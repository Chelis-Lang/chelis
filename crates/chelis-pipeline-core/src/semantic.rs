use chelis_types::infer::SignatureInferenceMetadata;
use chelis_types::{
    CheckedProgram, InferResult, TypeAnalysisOutcome, TypeEnv, analyze_ir_program,
    clean_fitness_from_stats,
};

use crate::roots::root_metadata;
use crate::{
    CheckedCompilation, PreparedProgram, PreparedTypeAnalysis, PreparedTypeAnalysisOutcome,
    SemanticContext, SemanticRejection,
};

/// Analyze prepared Deep through one monolithic IR type session.
pub fn analyze_prepared(prepared: PreparedProgram) -> PreparedTypeAnalysisOutcome {
    match analyze_ir_program(prepared.expanded_deep()) {
        TypeAnalysisOutcome::Rejected { fitness } => {
            PreparedTypeAnalysisOutcome::Rejected { fitness }
        }
        TypeAnalysisOutcome::Accepted { fitness, program } => {
            PreparedTypeAnalysisOutcome::Accepted(Box::new(PreparedTypeAnalysis {
                prepared,
                fitness,
                program: *program,
            }))
        }
    }
}

/// Analyze prepared Deep against a reusable library type context.
pub fn analyze_prepared_with_context(
    prepared: PreparedProgram,
    type_env: &TypeEnv,
    signature_context: &SignatureInferenceMetadata,
) -> Result<PreparedTypeAnalysis, InferResult> {
    let program = chelis_types::check_ir_with_signature_context(
        type_env,
        signature_context,
        prepared.expanded_deep(),
    )?;
    Ok(prepared_analysis_from_checked(prepared, program))
}

/// Adopt a checked type product from a specialized one-session context builder.
pub fn prepared_analysis_from_checked(
    prepared: PreparedProgram,
    program: CheckedProgram,
) -> PreparedTypeAnalysis {
    let fitness = clean_fitness_from_stats(
        chelis_types::structural_stats(prepared.expanded_deep()),
        program.infer_stats(),
    );
    PreparedTypeAnalysis {
        prepared,
        fitness,
        program,
    }
}

/// Complete effect and linearity checks in the canonical order.
pub fn complete_checks(
    analysis: PreparedTypeAnalysis,
    context: SemanticContext<'_>,
) -> Result<CheckedCompilation, SemanticRejection> {
    let effected = match context {
        SemanticContext::Isolated => chelis_effects::check_program(&analysis.program),
        SemanticContext::Library(library) => {
            chelis_effects::check_effects_with_context(library, &analysis.program)
        }
    }
    .map_err(|errors| SemanticRejection::Effects { errors })?;

    let program = match context {
        SemanticContext::Isolated => chelis_types::check_linearity(&effected),
        SemanticContext::Library(library) => {
            chelis_types::check_linearity_with_context(library, &effected)
        }
    }
    .map_err(|errors| SemanticRejection::Linearity { errors })?;

    let root_metadata = root_metadata(&program, None);
    Ok(CheckedCompilation {
        expanded_deep: analysis.prepared.expanded_deep,
        fitness: analysis.fitness,
        program,
        root_metadata,
    })
}

/// Compose a checked library and a checked extension for layered adapters.
///
/// # Safety
///
/// The library must have passed type, effect, and linearity checks. The extension
/// must have passed all three checks against that exact library.
#[doc(hidden)]
pub unsafe fn compose_checked(
    library: &CheckedProgram,
    extension: CheckedCompilation,
) -> CheckedCompilation {
    let (mut expanded_deep, _, extension_program, _) = extension.into_parts();
    let mut combined_deep = library.exprs().to_vec();
    combined_deep.append(&mut expanded_deep);
    let program = CheckedProgram::compose(library, &extension_program);
    let fitness = clean_fitness_from_stats(
        chelis_types::structural_stats(&combined_deep),
        program.infer_stats(),
    );
    let root_metadata = root_metadata(&program, None);
    CheckedCompilation {
        expanded_deep: combined_deep,
        fitness,
        program,
        root_metadata,
    }
}
