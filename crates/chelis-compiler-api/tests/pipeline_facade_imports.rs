use chelis_compiler_api::pipeline::{
    AllRootNames, CheckedCompilation, CheckedLibrary, ContextCheckedCompilation,
    ContextualLibraryTypeAnalysis, ContextualTypeAnalysis, ForwardNodeIndex, IrName,
    LibraryRejection, LoweredCompilation, LoweredLibrary, LoweredParts, LoweringMode, NamedRoots,
    PipelineGoal, PipelineOutcome, PipelineRejection, PipelineRequest, PreparationError,
    PreparedLibraryAnalysis, PreparedProgram, PreparedTypeAnalysis, PreparedTypeAnalysisOutcome,
    RootCountContext, RootMetadata, SemanticContext, SemanticRejection, TensorRootNames,
    analyze_prepared, analyze_prepared_library, analyze_prepared_library_with_base,
    analyze_prepared_with_library, check_prepared_library, complete_checks,
    complete_context_checks, complete_context_library_checks, complete_library_checks,
    lower_checked, lower_checked_with_context, lower_library, prepare_deep, prepare_source,
    prepare_surf_decls, run_prepared, run_source,
};

fn assert_type<T>() {}

fn assert_value<T>(_value: T) {}

#[test]
fn current_pipeline_surface_compiles_through_the_facade() {
    assert_type::<AllRootNames>();
    assert_type::<CheckedCompilation>();
    assert_type::<CheckedLibrary>();
    assert_type::<ContextCheckedCompilation<'static>>();
    assert_type::<ContextualLibraryTypeAnalysis<'static>>();
    assert_type::<ContextualTypeAnalysis<'static>>();
    assert_type::<ForwardNodeIndex>();
    assert_type::<IrName>();
    assert_type::<LibraryRejection>();
    assert_type::<LoweredCompilation>();
    assert_type::<LoweredLibrary>();
    assert_type::<LoweredParts>();
    assert_type::<LoweringMode>();
    assert_type::<NamedRoots>();
    assert_type::<PipelineGoal>();
    assert_type::<PipelineOutcome>();
    assert_type::<PipelineRejection>();
    assert_type::<PipelineRequest<'static>>();
    assert_type::<PreparationError>();
    assert_type::<PreparedLibraryAnalysis>();
    assert_type::<PreparedProgram>();
    assert_type::<PreparedTypeAnalysis>();
    assert_type::<PreparedTypeAnalysisOutcome>();
    assert_type::<RootCountContext>();
    assert_type::<RootMetadata>();
    assert_type::<SemanticContext>();
    assert_type::<SemanticRejection>();
    assert_type::<TensorRootNames>();

    assert_value(analyze_prepared);
    assert_value(analyze_prepared_library);
    assert_value(analyze_prepared_library_with_base);
    assert_value(analyze_prepared_with_library);
    assert_value(check_prepared_library);
    assert_value(complete_checks);
    assert_value(complete_context_checks);
    assert_value(complete_context_library_checks);
    assert_value(complete_library_checks);
    assert_value(lower_checked);
    assert_value(lower_checked_with_context);
    assert_value(lower_library);
    assert_value(prepare_deep);
    assert_value(prepare_source);
    assert_value(prepare_surf_decls);
    assert_value(run_prepared);
    assert_value(run_source);
}
