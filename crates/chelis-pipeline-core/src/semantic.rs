use chelis_types::infer::SignatureInferenceMetadata;
use chelis_types::{
    CheckedProgram, InferResult, TypeAnalysisOutcome, TypeEnv, analyze_ir_program,
    clean_fitness_from_stats,
};

use crate::roots::root_metadata;
use crate::{
    CheckedCompilation, CheckedLibrary, ContextCheckedCompilation, ContextualLibraryTypeAnalysis,
    ContextualTypeAnalysis, LibraryRejection, PreparedLibraryAnalysis, PreparedProgram,
    PreparedTypeAnalysis, PreparedTypeAnalysisOutcome, SemanticContext, SemanticRejection,
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

/// Analyze a prepared library and retain both products from one type session.
pub fn analyze_prepared_library(
    prepared: PreparedProgram,
) -> Result<PreparedLibraryAnalysis, LibraryRejection> {
    let (type_env, program) =
        chelis_types::build_compiled_library_context(prepared.expanded_deep())
            .map_err(|report| LibraryRejection::Type { report })?;
    let analysis = prepared_analysis_from_checked(prepared, program);
    Ok(PreparedLibraryAnalysis { type_env, analysis })
}

/// Complete semantic checks for a prepared library analysis.
pub fn complete_library_checks(
    prepared: PreparedLibraryAnalysis,
) -> Result<CheckedLibrary, LibraryRejection> {
    let checked = complete_checks(prepared.analysis, SemanticContext::Isolated)
        .map_err(library_semantic_rejection)?;
    bind_checked_library(prepared.type_env, checked)
}

/// Build a checked library and bind both products from its type session.
pub fn check_prepared_library(
    prepared: PreparedProgram,
) -> Result<CheckedLibrary, LibraryRejection> {
    complete_library_checks(analyze_prepared_library(prepared)?)
}

/// Bind cached library products whose lowering the caller re-derives and
/// compares.
///
/// The pair is accepted because its halves carry the same library proof
/// identity: the cached `TypeEnv` and `CheckedProgram` must agree on the
/// stored `LibraryProofId`, their declared-type maps must match, and the
/// `TypeEnv`'s exact callable-selector provenance must retain the digest bound
/// into the identity. That agreement is what rejects a forged pairing of two
/// individually well-formed halves.
///
/// The effect and linearity results the program carries are the producer's,
/// adopted rather than recomputed. That is sound only for a caller whose
/// payload also carries a lowering of the program, so only the
/// `CompiledContext` decoder and a `StdLibContext` decoder holding a stored
/// lowering call it. They establish the rest before or after this call: the envelope's payload digest, format
/// version and build identity pin the bytes to a producer running this
/// compiler build, the live source hash and cache identity pin them to the
/// sources being compiled, and the decoder re-lowers the program and compares
/// the result against the transmitted lowered payload, so an edit to either
/// half alone is rejected. The producer ran both checkers before writing, and
/// they are deterministic, so the adopted program is the one rerunning them
/// would produce; `chelis-compiler-api`'s
/// `cached_program_is_a_checker_fixed_point` locks that equality instead of
/// every load paying for it (chelis#2558). A payload with no lowering to
/// compare against goes through [`validate_cached_library`] instead.
///
/// The stored identity is not recomputed from the decoded source. It is bound
/// before the effect and linearity passes, and a layered build's cached library
/// is a composed program whose id is the extension's id over the base context
/// while its `exprs()` are the base++extension concatenation, so the id cannot
/// be reproduced from the decoded program's own source. Turning the id into a
/// source-recompute needs a canonical post-effects derivation; that is tracked
/// by the `canonicalize-library-proof-identity` OpenSpec change.
pub fn bind_cached_library(
    cached_type_env: TypeEnv,
    cached_program: CheckedProgram,
) -> Result<CheckedLibrary, LibraryRejection> {
    if !cached_type_env.matches_checked_program(&cached_program) {
        return Err(LibraryRejection::ContextMismatch);
    }
    Ok(CheckedLibrary {
        type_env: cached_type_env,
        program: cached_program,
    })
}

/// Bind cached library products that carry no lowering, rerunning the effect
/// and linearity checkers over the program.
///
/// This is the `LibraryContext` (dependency typecheck cache) decoder's route,
/// and a `StdLibContext` decoder's when its optional lowering is absent. Such
/// a payload carries the type environment and the checked program and nothing
/// derived from them, so an edit confined to the program under a recomputed
/// payload digest has no second derivation to disagree with. The reruns are
/// what reject such a program when the edit breaks an effect or linearity
/// rule, and they replace the stored annotations with freshly derived ones
/// (chelis#2558). The proof-identity agreement is the one
/// [`bind_cached_library`] checks, before and after the reruns.
pub fn validate_cached_library(
    cached_type_env: TypeEnv,
    cached_program: CheckedProgram,
) -> Result<CheckedLibrary, LibraryRejection> {
    if !cached_type_env.matches_checked_program(&cached_program) {
        return Err(LibraryRejection::ContextMismatch);
    }
    let effected = chelis_effects::check_program(&cached_program)
        .map_err(|errors| LibraryRejection::Effects { errors })?;
    let program = chelis_types::check_linearity(&effected)
        .map_err(|errors| LibraryRejection::Linearity { errors })?;
    if !cached_type_env.matches_checked_program(&program) {
        return Err(LibraryRejection::ContextMismatch);
    }
    Ok(CheckedLibrary {
        type_env: cached_type_env,
        program,
    })
}

/// Analyze prepared Deep against a reusable library type context.
fn analyze_prepared_with_context(
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

/// Analyze prepared Deep against one exact checked library.
pub fn analyze_prepared_with_library<'library>(
    prepared: PreparedProgram,
    library: &'library CheckedLibrary,
) -> Result<ContextualTypeAnalysis<'library>, InferResult> {
    let analysis = analyze_prepared_with_context(
        prepared,
        library.type_env(),
        library.program().signature_inference(),
    )?;
    Ok(ContextualTypeAnalysis { library, analysis })
}

/// Analyze a library extension on top of one exact checked base library.
pub fn analyze_prepared_library_with_base<'library>(
    prepared: PreparedProgram,
    library: &'library CheckedLibrary,
) -> Result<ContextualLibraryTypeAnalysis<'library>, InferResult> {
    let (type_env, program) = chelis_types::build_compiled_library_context_with_base(
        library.type_env(),
        prepared.expanded_deep(),
    )?;
    let analysis = prepared_analysis_from_checked(prepared, program);
    Ok(ContextualLibraryTypeAnalysis {
        library,
        type_env,
        analysis,
    })
}

fn prepared_analysis_from_checked(
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

/// Complete isolated effect and linearity checks in the canonical order.
pub fn complete_checks(
    analysis: PreparedTypeAnalysis,
    context: SemanticContext,
) -> Result<CheckedCompilation, SemanticRejection> {
    match context {
        SemanticContext::Isolated => complete_checks_in_context(analysis, None),
    }
}

fn complete_checks_in_context(
    analysis: PreparedTypeAnalysis,
    library: Option<&CheckedProgram>,
) -> Result<CheckedCompilation, SemanticRejection> {
    let effected = if let Some(library) = library {
        chelis_effects::check_effects_with_context(library, &analysis.program)
    } else {
        chelis_effects::check_program(&analysis.program)
    }
    .map_err(|errors| SemanticRejection::Effects { errors })?;

    let program = if let Some(library) = library {
        chelis_types::check_linearity_with_context(library, &effected)
    } else {
        chelis_types::check_linearity(&effected)
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

/// Complete semantic checks against the library bound to the analysis.
pub fn complete_context_checks<'library>(
    contextual: ContextualTypeAnalysis<'library>,
) -> Result<ContextCheckedCompilation<'library>, SemanticRejection> {
    let extension =
        complete_checks_in_context(contextual.analysis, Some(contextual.library.program()))?;
    Ok(ContextCheckedCompilation {
        library: contextual.library,
        extension,
    })
}

/// Complete checks and compose one library extension with its bound base.
pub fn complete_context_library_checks(
    contextual: ContextualLibraryTypeAnalysis<'_>,
) -> Result<CheckedLibrary, LibraryRejection> {
    let extension =
        complete_checks_in_context(contextual.analysis, Some(contextual.library.program()))
            .map_err(library_semantic_rejection)?;
    let checked = compose_context(contextual.library, extension);
    bind_checked_library(contextual.type_env, checked)
}

impl ContextCheckedCompilation<'_> {
    /// Compose this extension with the exact library that produced it.
    pub fn compose(self) -> CheckedCompilation {
        compose_context(self.library, self.extension)
    }
}

fn compose_context(library: &CheckedLibrary, extension: CheckedCompilation) -> CheckedCompilation {
    let (mut expanded_deep, _, extension_program, _) = extension.into_parts();
    let mut combined_deep = library.program().exprs().to_vec();
    combined_deep.append(&mut expanded_deep);
    let program = CheckedProgram::compose(library.program(), &extension_program)
        .expect("context-checked extensions retain their exact library proof");
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

fn bind_checked_library(
    type_env: TypeEnv,
    checked: CheckedCompilation,
) -> Result<CheckedLibrary, LibraryRejection> {
    let (_, _, program, _) = checked.into_parts();
    if !type_env.matches_checked_program(&program) {
        return Err(LibraryRejection::ContextMismatch);
    }
    Ok(CheckedLibrary { type_env, program })
}

fn library_semantic_rejection(rejection: SemanticRejection) -> LibraryRejection {
    match rejection {
        SemanticRejection::Effects { errors } => LibraryRejection::Effects { errors },
        SemanticRejection::Linearity { errors } => LibraryRejection::Linearity { errors },
    }
}
