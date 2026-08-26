use chelis_types::{TypeAnalysisOutcome, analyze_ir_program};

fn deep(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parser::parse_str(source).expect("Deep fixture must parse")
}

#[test]
fn accepted_analysis_carries_fitness_and_the_checked_program() {
    let exprs = deep("(def {} answer (lit {type: (t-prim {} int32)} 42))");

    let TypeAnalysisOutcome::Accepted { fitness, program } = analyze_ir_program(&exprs) else {
        panic!("valid Deep must produce an accepted type analysis");
    };

    assert!(fitness.errors.is_empty());
    assert_eq!(fitness.typed_nodes, program.infer_stats().typed_nodes);
    assert_eq!(fitness.total_nodes, program.infer_stats().total_nodes);
}

#[test]
fn rejected_analysis_carries_fitness_without_a_checked_program() {
    let exprs = deep("(def {} answer (var {} missing))");

    let TypeAnalysisOutcome::Rejected { fitness } = analyze_ir_program(&exprs) else {
        panic!("an unbound name must reject type analysis");
    };

    assert!(!fitness.errors.is_empty());
    assert!(fitness.score < 1.0);
    assert!(fitness.unresolved_names.contains(&"missing".to_string()));
}

#[test]
fn analysis_rejects_a_top_level_binding_cycle() {
    let surf = "module Cycle\na: int32 = add(b, 1)\nb: int32 = add(a, 1)\n";
    let decls = chelis_surf::parser::parse_str(surf).expect("Surf fixture must parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);

    let TypeAnalysisOutcome::Rejected { fitness } = analyze_ir_program(&exprs) else {
        panic!("a top-level binding cycle must reject type analysis");
    };

    assert!(
        fitness
            .errors
            .iter()
            .any(|error| error.message.contains("cycle")),
        "the rejection must identify the cycle: {:?}",
        fitness.errors
    );
}

#[test]
fn analysis_accepts_finite_recursive_functions() {
    let surf =
        "module Rec\ndef descend(n: int32) -> int32 = if eq(n, 0) then 0 else descend(sub(n, 1))\n";
    let decls = chelis_surf::parser::parse_str(surf).expect("Surf fixture must parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);

    assert!(matches!(
        analyze_ir_program(&exprs),
        TypeAnalysisOutcome::Accepted { .. }
    ));
}

#[test]
fn analysis_rejects_recursion_without_a_base_case() {
    let surf = "module Rec\ndef forever(n: int32) -> int32 = forever(n)\n";
    let decls = chelis_surf::parser::parse_str(surf).expect("Surf fixture must parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);

    let TypeAnalysisOutcome::Rejected { fitness } = analyze_ir_program(&exprs) else {
        panic!("base-case-free recursion must reject type analysis");
    };
    assert!(
        fitness
            .errors
            .iter()
            .any(|error| error.message.contains("no base case")),
        "the rejection must identify the missing base case: {:?}",
        fitness.errors
    );
}

#[test]
fn analysis_accepts_a_deep_finite_expression() {
    let mut body = "0".to_string();
    for _ in 0..32 {
        body = format!("add({body}, 1)");
    }
    let surf = format!("module Deep\ndef value() -> int32 = {body}\n");
    let decls = chelis_surf::parser::parse_str(&surf).expect("deep Surf fixture must parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);

    assert!(matches!(
        analyze_ir_program(&exprs),
        TypeAnalysisOutcome::Accepted { .. }
    ));
}
