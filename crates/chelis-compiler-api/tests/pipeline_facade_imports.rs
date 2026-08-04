use chelis_compiler_api::pipeline::{
    AllRootNames, CheckedCompilation, ForwardNodeIndex, IrName, LoweredCompilation, LoweredParts,
    LoweringMode, NamedRoots, PipelineGoal, PipelineOutcome, PipelineRejection, PipelineRequest,
    PreparationError, PreparedProgram, PreparedTypeAnalysis, PreparedTypeAnalysisOutcome,
    RootCountContext, RootMetadata, SemanticContext, SemanticRejection, TensorRootNames,
    analyze_prepared, analyze_prepared_with_context, complete_checks, lower_checked,
    lower_checked_with_context, lower_library, prepare_deep, prepare_source, prepare_surf_decls,
    prepared_analysis_from_checked, run_prepared, run_source,
};

fn assert_type<T>() {}

fn assert_value<T>(_value: T) {}

#[test]
fn current_pipeline_surface_compiles_through_the_facade() {
    assert_type::<AllRootNames>();
    assert_type::<CheckedCompilation>();
    assert_type::<ForwardNodeIndex>();
    assert_type::<IrName>();
    assert_type::<LoweredCompilation>();
    assert_type::<LoweredParts>();
    assert_type::<LoweringMode>();
    assert_type::<NamedRoots>();
    assert_type::<PipelineGoal>();
    assert_type::<PipelineOutcome>();
    assert_type::<PipelineRejection>();
    assert_type::<PipelineRequest<'static>>();
    assert_type::<PreparationError>();
    assert_type::<PreparedProgram>();
    assert_type::<PreparedTypeAnalysis>();
    assert_type::<PreparedTypeAnalysisOutcome>();
    assert_type::<RootCountContext>();
    assert_type::<RootMetadata>();
    assert_type::<SemanticContext<'static>>();
    assert_type::<SemanticRejection>();
    assert_type::<TensorRootNames>();

    assert_value(analyze_prepared);
    assert_value(analyze_prepared_with_context);
    assert_value(complete_checks);
    assert_value(lower_checked);
    assert_value(lower_checked_with_context);
    assert_value(lower_library);
    assert_value(prepare_deep);
    assert_value(prepare_source);
    assert_value(prepare_surf_decls);
    assert_value(prepared_analysis_from_checked);
    assert_value(run_prepared);
    assert_value(run_source);
}
