use chelis_types::{TypeAnalysisOutcome, analyze_ir_program};

fn deep(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parser::parse_str(source).expect("Deep fixture must parse")
}

#[test]
fn accepted_analysis_carries_fitness_and_the_checked_program() {
    let exprs = deep("(def {} answer (lit {type: (t-prim {} i32)} 42))");

    let TypeAnalysisOutcome::Accepted { fitness, program } =
        analyze_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked)
    else {
        panic!("valid Deep must produce an accepted type analysis");
    };

    assert!(fitness.errors.is_empty());
    assert_eq!(fitness.typed_nodes, program.infer_stats().typed_nodes);
    assert_eq!(fitness.total_nodes, program.infer_stats().total_nodes);
}

#[test]
fn rejected_analysis_carries_fitness_without_a_checked_program() {
    let exprs = deep("(def {} answer (var {} missing))");

    let TypeAnalysisOutcome::Rejected { fitness } =
        analyze_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked)
    else {
        panic!("an unbound name must reject type analysis");
    };

    assert!(!fitness.errors.is_empty());
    assert!(fitness.score < 1.0);
    assert!(fitness.unresolved_names.contains(&"missing".to_string()));
}

#[test]
fn analysis_rejects_a_top_level_binding_cycle() {
    let exprs = deep(include_str!(
        "fixtures/core/tests/type_analysis_outcome/case_001.deep"
    ));

    let TypeAnalysisOutcome::Rejected { fitness } =
        analyze_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked)
    else {
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
    let exprs = deep(include_str!(
        "fixtures/core/tests/type_analysis_outcome/case_002.deep"
    ));

    assert!(matches!(
        analyze_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked),
        TypeAnalysisOutcome::Accepted { .. }
    ));
}

#[test]
fn analysis_accepts_recursion_without_a_base_case() {
    let exprs = deep(include_str!(
        "fixtures/core/tests/type_analysis_outcome/case_003.deep"
    ));

    // [04-INF-2]/[04-INF-3] admit uniform recursion. A backend may still refuse
    // an unsupported recursive lowering under chelis#730, but that capability
    // boundary is not a type-analysis rejection.
    assert!(matches!(
        analyze_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked),
        TypeAnalysisOutcome::Accepted { .. }
    ));
}

#[test]
fn analysis_accepts_a_deep_finite_expression() {
    let mut body = "(lit {type: (t-prim {} i32)} 0)".to_string();
    for _ in 0..32 {
        body = format!("(app {{}} (var {{}} add) {body} (lit {{type: (t-prim {{}} i32)}} 1))");
    }
    let source = include_str!("fixtures/core/tests/type_analysis_outcome/finite_depth.deep")
        .replace("@@body@@", &body);
    let exprs = deep(&source);

    assert!(matches!(
        analyze_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked),
        TypeAnalysisOutcome::Accepted { .. }
    ));
}
