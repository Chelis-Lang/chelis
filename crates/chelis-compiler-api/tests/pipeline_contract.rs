use chelis_compiler_api::pipeline::{
    LoweringMode, PipelineGoal, PipelineOutcome, PipelineRejection, PipelineRequest, run_source,
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

#[test]
fn type_analysis_goal_returns_fitness_and_one_checked_product() {
    let outcome = run_source(request(
        "def answer -> int32 = 42\n",
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
        "def identity(x: tensor[n, f32]) -> tensor[n, f32] = x\n",
        PipelineGoal::FullCheck,
    ))
    .expect("valid source must pass all semantic stages");

    let PipelineOutcome::Checked(checked) = outcome else {
        panic!("the full-check goal must return CheckedCompilation");
    };
    assert!(checked.fitness().errors.is_empty());
    assert_eq!(checked.root_metadata().all_names(), &["identity"]);
}

#[test]
fn lower_goal_returns_checked_state_dag_and_canonical_tuple_roots() {
    let source = "def pair(x: tensor[n, f32]) -> (tensor[n, f32], tensor[n, f32]) = (copy(x), x)\n";
    let outcome = run_source(request(source, PipelineGoal::Lower(LoweringMode::Strict)))
        .expect("valid tuple source must lower");

    let PipelineOutcome::Lowered(lowered) = outcome else {
        panic!("the lower goal must return LoweredCompilation");
    };
    assert_eq!(
        lowered.checked().root_metadata().tensor_names(),
        &["pair.0", "pair.1"]
    );
    assert_eq!(lowered.named_roots().len(), 2);
    assert_eq!(lowered.dag().roots().len(), 2);
}

#[test]
fn parse_rejection_stops_before_type_analysis() {
    let rejection = run_source(request("def broken(\n", PipelineGoal::FullCheck))
        .expect_err("invalid Surf must reject during preparation");
    assert!(matches!(rejection, PipelineRejection::Preparation(_)));
}

#[test]
fn type_rejection_has_no_checked_or_lowered_product() {
    let rejection = run_source(request(
        "def broken -> int32 = missing\n",
        PipelineGoal::Lower(LoweringMode::Strict),
    ))
    .expect_err("an unbound name must reject type analysis");
    let PipelineRejection::Type { fitness } = rejection else {
        panic!("the rejection must retain its type stage");
    };
    assert!(!fitness.errors.is_empty());
}

#[test]
fn effect_rejection_retains_its_native_error_stage() {
    let rejection = run_source(request(
        "def noisy(x: tensor[4, f32]) -> tensor[4, f32] ! { } = dropout(x, 0.5)\n",
        PipelineGoal::FullCheck,
    ))
    .expect_err("a pure declaration cannot perform Random");
    assert!(matches!(rejection, PipelineRejection::Effects { .. }));
}

#[test]
fn linearity_rejection_retains_its_native_error_stage() {
    let rejection = run_source(request(
        "def broken(x: tensor[4, f32]) -> tensor[4, f32] = { y = realize(x); add(x, y) }\n",
        PipelineGoal::FullCheck,
    ))
    .expect_err("a consumed tensor cannot be used again");
    assert!(matches!(rejection, PipelineRejection::Linearity { .. }));
}

#[test]
fn strict_lowering_rejection_retains_the_lower_diagnostic() {
    let source = "def loss(theta: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(floor(copy(theta)), 0))\ngrad_loss = grad(loss, wrt=(theta))\nout = grad_loss(to_tensor([1.5, 2.5]))\n";
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
