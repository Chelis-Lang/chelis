use chelis_compiler_api::pipeline::{
    IrName, LoweringMode, PipelineGoal, PipelineOutcome, PipelineRejection, PipelineRequest,
    PreparedTypeAnalysis, PreparedTypeAnalysisOutcome, SemanticContext, SemanticRejection,
    analyze_prepared, check_prepared_library, complete_checks, lower_checked, lower_library,
    prepare_source, run_prepared, run_source,
};
use chelis_compiler_api::schema::SourceKind;
use chelis_types::TypeAnalysisOutcome;

fn request(source: &str, goal: PipelineGoal) -> PipelineRequest<'_> {
    PipelineRequest {
        source_kind: SourceKind::Surf,
        source,
        entry: None,
        goal,
    }
}

fn accepted_analysis(source: &str) -> PreparedTypeAnalysis {
    let prepared = prepare_source(SourceKind::Surf, source, None).expect("source must prepare");
    match analyze_prepared(prepared) {
        PreparedTypeAnalysisOutcome::Accepted(analysis) => *analysis,
        PreparedTypeAnalysisOutcome::Rejected { fitness } => {
            panic!("source must pass type analysis: {:?}", fitness.errors)
        }
    }
}

fn root_name_text<'a>(names: impl Iterator<Item = &'a IrName>) -> Vec<&'a str> {
    names.map(IrName::as_str).collect()
}

#[test]
fn type_analysis_goal_returns_fitness_and_one_checked_product() {
    let outcome = run_source(request(
        "def answer() -> i32 = 42\n",
        PipelineGoal::TypeAnalysis,
    ))
    .expect("valid source must prepare");

    let PipelineOutcome::TypeAnalysis(TypeAnalysisOutcome::Accepted { fitness, program }) = outcome
    else {
        panic!("the type-analysis goal must stop after type analysis");
    };
    assert!(fitness.errors.is_empty());
    assert_eq!(fitness.total_nodes, program.infer_stats().total_nodes);
}

#[test]
fn full_check_goal_returns_a_fully_checked_state() {
    let outcome = run_source(request(
        "def identity[n](x: tensor[n, f32]) -> tensor[n, f32] = x\n",
        PipelineGoal::FullCheck,
    ))
    .expect("valid source must pass all semantic stages");

    let PipelineOutcome::Checked(checked) = outcome else {
        panic!("the full-check goal must return CheckedCompilation");
    };
    assert!(checked.fitness().errors.is_empty());
    assert_eq!(
        root_name_text(checked.root_metadata().all_names().iter()),
        ["identity"]
    );
}

#[test]
fn lower_goal_returns_checked_state_dag_and_canonical_tuple_roots() {
    let source =
        "def pair[n](x: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (copy(x), x)\n";
    let outcome = run_source(request(source, PipelineGoal::Lower(LoweringMode::Strict)))
        .expect("valid tuple source must lower");

    let PipelineOutcome::Lowered(lowered) = outcome else {
        panic!("the lower goal must return LoweredCompilation");
    };
    assert_eq!(
        root_name_text(lowered.checked().root_metadata().tensor_names().iter()),
        ["pair.0", "pair.1"]
    );
    assert_eq!(lowered.named_roots().len(), 2);
    assert_eq!(lowered.dag().roots().len(), 2);

    let parts = lowered.into_parts();
    assert_eq!(parts.named_roots.len(), 2);
    assert_eq!(parts.forward_node_index.len(), 3);
    assert_eq!(parts.dag.roots().len(), 2);
    assert!(parts.checked.fitness().errors.is_empty());
}

#[test]
fn declared_roots_and_forward_load_aliases_have_distinct_lookups() {
    let outcome = run_source(request(
        "def out[n](input: tensor[n, f32]) -> tensor[n, f32] = relu(input)\n",
        PipelineGoal::Lower(LoweringMode::Strict),
    ))
    .expect("valid source must lower");
    let PipelineOutcome::Lowered(lowered) = outcome else {
        panic!("the lower goal must return LoweredCompilation");
    };

    let output_name = IrName::new("out");
    let input_name = IrName::new("input");
    assert!(lowered.named_roots().get(&output_name).is_some());
    assert!(lowered.named_roots().get(&input_name).is_none());
    assert!(lowered.forward_node_index().get(&output_name).is_some());
    assert!(lowered.forward_node_index().get(&input_name).is_some());
}

#[test]
fn host_only_output_keeps_the_explicit_empty_root_product() {
    let outcome = run_source(request(
        "label = \"host only\"\n",
        PipelineGoal::Lower(LoweringMode::AllowHostOnly),
    ))
    .expect("host-only source must preserve the empty DAG fallback");
    let PipelineOutcome::Lowered(lowered) = outcome else {
        panic!("the lower goal must return LoweredCompilation");
    };

    assert!(lowered.dag().roots().is_empty());
    assert!(lowered.named_roots().is_empty());
}

#[test]
fn parse_rejection_stops_before_type_analysis() {
    let rejection = run_source(request("def broken(\n", PipelineGoal::FullCheck))
        .expect_err("invalid Surf must reject during preparation");
    assert!(matches!(rejection, PipelineRejection::Preparation(_)));
}

#[test]
fn pre_cancelled_pipeline_rejects_structurally_before_parsing() {
    let token = chelis_types::CancelToken::new();
    token.cancel();
    let _guard = chelis_types::install_cancel_token(token);

    let rejection = run_source(request(
        "def answer() -> i32 = 42\n",
        PipelineGoal::FullCheck,
    ))
    .expect_err("a pre-cancelled pipeline must not start parsing");

    assert!(matches!(
        rejection,
        PipelineRejection::Cancelled { stage: "parse" }
    ));
}

#[test]
fn direct_lower_functions_keep_the_lower_cancellation_stage() {
    let checked = complete_checks(
        accepted_analysis("def identity[n](x: tensor[n, f32]) -> tensor[n, f32] = x\n"),
        SemanticContext::Isolated,
    )
    .expect("the fixture must pass semantic checks");
    let library = check_prepared_library(
        prepare_source(SourceKind::Surf, "def library_value() -> i32 = 1\n", None)
            .expect("the library source must prepare"),
    )
    .expect("the library fixture must pass semantic checks");

    let token = chelis_types::CancelToken::new();
    token.cancel();
    let _guard = chelis_types::install_cancel_token(token);

    let checked_rejection = lower_checked(checked, LoweringMode::Strict)
        .expect_err("direct checked lowering must observe cancellation");
    assert!(matches!(
        checked_rejection,
        PipelineRejection::Cancelled { stage: "lower" }
    ));

    let library_rejection =
        lower_library(&library).expect_err("direct library lowering must observe cancellation");
    assert!(matches!(
        library_rejection,
        PipelineRejection::Cancelled { stage: "lower" }
    ));
}

#[test]
fn dynamic_pipeline_goals_keep_their_initial_cancellation_stage() {
    let prepared = [
        PipelineGoal::TypeAnalysis,
        PipelineGoal::FullCheck,
        PipelineGoal::Lower(LoweringMode::Strict),
    ]
    .map(|goal| {
        (
            goal,
            prepare_source(SourceKind::Surf, "def answer() -> i32 = 42\n", None)
                .expect("the fixture source must prepare"),
        )
    });
    let token = chelis_types::CancelToken::new();
    token.cancel();
    let _guard = chelis_types::install_cancel_token(token);

    for (goal, prepared) in prepared {
        let rejection = run_prepared(prepared, goal)
            .expect_err("every dynamic goal must observe cancellation before analysis");
        assert!(matches!(
            rejection,
            PipelineRejection::Cancelled { stage: "check" }
        ));
    }
}

#[test]
fn type_rejection_has_no_checked_or_lowered_product() {
    let rejection = run_source(request(
        "def broken() -> i32 = missing\n",
        PipelineGoal::Lower(LoweringMode::Strict),
    ))
    .expect_err("an unbound name must reject type analysis");
    let PipelineRejection::Type { fitness } = rejection else {
        panic!("the rejection must retain its type stage");
    };
    assert!(!fitness.errors.is_empty());
}

#[test]
fn complete_checks_returns_a_clean_checked_product() {
    let checked = complete_checks(
        accepted_analysis("def identity[n](x: tensor[n, f32]) -> tensor[n, f32] = x\n"),
        SemanticContext::Isolated,
    )
    .expect("clean semantics must return the checked product");
    assert!(checked.fitness().errors.is_empty());
}

#[test]
fn effect_rejection_uses_the_narrow_semantic_error() {
    let source = "def noisy(x: tensor[4, f32]) -> tensor[4, f32] ! { } = debug(x)\n";
    let rejection = complete_checks(accepted_analysis(source), SemanticContext::Isolated)
        .expect_err("a pure declaration cannot perform IO");
    assert!(matches!(rejection, SemanticRejection::Effects { .. }));

    let full_rejection = run_source(request(source, PipelineGoal::FullCheck))
        .expect_err("the full pipeline must retain the effect stage");
    assert!(matches!(full_rejection, PipelineRejection::Effects { .. }));
}

#[test]
fn linearity_rejection_uses_the_narrow_semantic_error() {
    let source =
        "def broken(x: tensor[4, f32]) -> tensor[4, f32] = {\n  y = realize(x)\n  add(x, y)\n}\n";
    let rejection = complete_checks(accepted_analysis(source), SemanticContext::Isolated)
        .expect_err("a consumed tensor cannot be used again");
    assert!(matches!(rejection, SemanticRejection::Linearity { .. }));

    let full_rejection = run_source(request(source, PipelineGoal::FullCheck))
        .expect_err("the full pipeline must retain the linearity stage");
    assert!(matches!(
        full_rejection,
        PipelineRejection::Linearity { .. }
    ));
}

#[test]
fn strict_lowering_rejection_retains_the_lower_diagnostic() {
    let source = "def loss(theta: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(floor(copy(theta)), 0))\ngrad_loss = grad(loss, wrt=theta)\nout = grad_loss(to_tensor([1.5, 2.5]))\n";
    let rejection = run_source(request(source, PipelineGoal::Lower(LoweringMode::Strict)))
        .expect_err("grad through floor must reject during lowering");
    let PipelineRejection::Lower(diagnostic) = rejection else {
        panic!("the rejection must retain its lower stage");
    };
    assert!(
        diagnostic.to_string().contains("grad"),
        "unexpected lower diagnostic: {diagnostic}"
    );
}
