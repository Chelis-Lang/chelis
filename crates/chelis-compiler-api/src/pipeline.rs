//! Canonical compiler front-end facade.
//!
//! The dependency-bottom core owns semantic transitions. This module owns source
//! preparation, dynamic goal selection, cancellation, and host policy selection.
//!
//! A rejection has no success-product accessor:
//!
//! ```compile_fail
//! use chelis_compiler_api::pipeline::PipelineRejection;
//! fn expose_checked(rejection: PipelineRejection) {
//!     let _ = rejection.checked();
//! }
//! ```
//!
//! Pipeline state is not a machine-facing wire model:
//!
//! ```
//! fn require_serialize<T: serde::Serialize>() {}
//! require_serialize::<chelis_compiler_api::schema::CheckResult>();
//! ```
//!
//! ```compile_fail
//! fn require_serialize<T: serde::Serialize>() {}
//! require_serialize::<chelis_compiler_api::pipeline::CheckedCompilation>();
//! ```
//!
//! Checked and lowered libraries are not cache or wire models:
//!
//! ```compile_fail
//! fn require_serialize<T: serde::Serialize>() {}
//! require_serialize::<chelis_compiler_api::pipeline::CheckedLibrary>();
//! require_serialize::<chelis_compiler_api::pipeline::LoweredLibrary>();
//! ```
//!
//! The facade exports no unchecked proof-adoption helper:
//!
//! ```compile_fail
//! use chelis_compiler_api::pipeline::prepared_analysis_from_checked;
//!
//! let _ = prepared_analysis_from_checked;
//! ```
//!
//! The facade exports no raw library binding constructor:
//!
//! ```compile_fail
//! use chelis_compiler_api::pipeline::bind_checked_library;
//!
//! let _ = bind_checked_library;
//! ```
//!
//! Declared roots and the forward node index are different products:
//!
//! ```compile_fail
//! use chelis_compiler_api::pipeline::{LoweredParts, NamedRoots};
//!
//! fn consume_declared_roots(_: NamedRoots) {}
//!
//! let parts: LoweredParts = todo!();
//! consume_declared_roots(parts.forward_node_index);
//! ```

use std::fmt;

use chelis_deep::Expr as DeepExpr;
use chelis_ir::lower::LowerDiagnostic;
use chelis_pipeline_core::CoreLowerError;
use chelis_types::{FitnessReport, TypeAnalysisOutcome};

use crate::schema::SourceKind;

pub use chelis_pipeline_core::{
    AllRootNames, CheckedCompilation, CheckedLibrary, ContextCheckedCompilation,
    ContextualLibraryTypeAnalysis, ContextualTypeAnalysis, ForwardNodeIndex, IrName,
    LibraryRejection, LoweredCompilation, LoweredLibrary, LoweredParts, LoweringMode, NamedRoots,
    PreparedLibraryAnalysis, PreparedProgram, PreparedTypeAnalysis, PreparedTypeAnalysisOutcome,
    RootCountContext, RootMetadata, SemanticContext, SemanticRejection, TensorRootNames,
    analyze_prepared, analyze_prepared_library, analyze_prepared_library_with_base,
    analyze_prepared_with_library, check_prepared_library, complete_checks,
    complete_context_checks, complete_context_library_checks, complete_library_checks,
    lower_checked_for_evaluation, lower_checked_with_evaluation_context,
};

/// The closed set of supported pipeline goals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineGoal {
    /// Stop after type analysis and fitness production.
    TypeAnalysis,
    /// Run type, effect, and linearity checks.
    FullCheck,
    /// Run full checks and lower the checked program.
    Lower(LoweringMode),
}

/// A source request for the canonical pipeline.
#[derive(Debug, Clone, Copy)]
pub struct PipelineRequest<'a> {
    pub source_kind: SourceKind,
    pub source: &'a str,
    pub entry: Option<&'a str>,
    pub goal: PipelineGoal,
}

/// Typed preparation failures. Formatting stays at each consumer boundary.
#[derive(Debug)]
pub enum PreparationError {
    SurfParse {
        source: String,
        error: chelis_surf::parser::ParseError,
    },
    /// A Deep text ingress rejection: a lex/parse failure, or a role-stamp
    /// failure that means the text never denoted a well-formed AST
    /// (chelis#1088).
    DeepParse(chelis_deep::StampOrParseError),
    Expansion(chelis_macros::ExpansionError),
}

impl PreparationError {
    pub fn surf_source(&self) -> Option<&str> {
        match self {
            Self::SurfParse { source, .. } => Some(source),
            Self::DeepParse(_) | Self::Expansion(_) => None,
        }
    }
}

impl fmt::Display for PreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SurfParse { error, .. } => write!(formatter, "{error}"),
            Self::DeepParse(error) => write!(formatter, "{error}"),
            Self::Expansion(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for PreparationError {}

/// Native typed failures from canonical preparation, semantic, and lowering stages.
#[derive(Debug)]
pub enum PipelineRejection {
    Cancelled {
        stage: &'static str,
    },
    Preparation(PreparationError),
    Type {
        fitness: FitnessReport,
    },
    Effects {
        errors: Vec<chelis_effects::EffectError>,
    },
    Linearity {
        errors: Vec<chelis_types::errors::CheckError>,
    },
    Lower(LowerDiagnostic),
    RootCount {
        context: RootCountContext,
        expected: usize,
        actual: usize,
    },
}

impl From<SemanticRejection> for PipelineRejection {
    fn from(rejection: SemanticRejection) -> Self {
        match rejection {
            SemanticRejection::Effects { errors } => Self::Effects { errors },
            SemanticRejection::Linearity { errors } => Self::Linearity { errors },
        }
    }
}

impl From<CoreLowerError> for PipelineRejection {
    fn from(error: CoreLowerError) -> Self {
        match error {
            CoreLowerError::Lower(diagnostic) => Self::Lower(diagnostic),
            CoreLowerError::RootCount {
                context,
                expected,
                actual,
            } => Self::RootCount {
                context,
                expected,
                actual,
            },
        }
    }
}

impl fmt::Display for PipelineRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled { .. } => formatter.write_str(chelis_types::EVAL_CANCELLED_MSG),
            Self::Preparation(error) => write!(formatter, "{error}"),
            Self::Type { fitness } => write!(formatter, "Type errors: {:?}", fitness.errors),
            Self::Effects { errors } => {
                write_joined_messages(formatter, errors.iter().map(|error| error.message.as_str()))
            }
            Self::Linearity { errors } => {
                write_joined_messages(formatter, errors.iter().map(|error| error.message.as_str()))
            }
            Self::Lower(diagnostic) => write!(formatter, "{diagnostic}"),
            Self::RootCount {
                context,
                expected,
                actual,
            } => {
                let subject = match context {
                    RootCountContext::Program => "root",
                    RootCountContext::NewCode => "new-code root",
                };
                write!(
                    formatter,
                    "lowered {subject} count mismatch: expected {expected} named roots, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for PipelineRejection {}

fn write_joined_messages<'a>(
    formatter: &mut fmt::Formatter<'_>,
    messages: impl Iterator<Item = &'a str>,
) -> fmt::Result {
    for (index, message) in messages.enumerate() {
        if index > 0 {
            formatter.write_str("; ")?;
        }
        formatter.write_str(message)?;
    }
    Ok(())
}

/// A successful result for the requested goal.
#[derive(Debug)]
pub enum PipelineOutcome {
    TypeAnalysis(TypeAnalysisOutcome),
    Checked(CheckedCompilation),
    Lowered(LoweredCompilation),
}

/// Run a source request through the required pipeline prefix.
pub fn run_source(request: PipelineRequest<'_>) -> Result<PipelineOutcome, PipelineRejection> {
    pipeline_bail_if_cancelled("parse")?;
    let prepared = prepare_source(request.source_kind, request.source, request.entry)
        .map_err(PipelineRejection::Preparation)?;
    pipeline_bail_if_cancelled("check")?;
    run_prepared(prepared, request.goal)
}

/// Run a prepared program through the required pipeline prefix.
pub fn run_prepared(
    prepared: PreparedProgram,
    goal: PipelineGoal,
) -> Result<PipelineOutcome, PipelineRejection> {
    match goal {
        PipelineGoal::TypeAnalysis => {
            pipeline_bail_if_cancelled("check")?;
            let analysis = match analyze_prepared(prepared) {
                PreparedTypeAnalysisOutcome::Rejected { fitness } => {
                    TypeAnalysisOutcome::Rejected { fitness }
                }
                PreparedTypeAnalysisOutcome::Accepted(analysis) => {
                    let (_, fitness, program) = analysis.into_parts();
                    TypeAnalysisOutcome::Accepted {
                        fitness,
                        program: Box::new(program),
                    }
                }
            };
            pipeline_bail_if_cancelled("check")?;
            Ok(PipelineOutcome::TypeAnalysis(analysis))
        }
        PipelineGoal::FullCheck => {
            let analysis = require_accepted_analysis(prepared)?;
            pipeline_bail_if_cancelled("effects")?;
            complete_checks(analysis, SemanticContext::Isolated)
                .map(PipelineOutcome::Checked)
                .map_err(lift_semantic_rejection)
                .and_then(|outcome| {
                    pipeline_bail_if_cancelled("linearity")?;
                    Ok(outcome)
                })
        }
        PipelineGoal::Lower(mode) => {
            let analysis = require_accepted_analysis(prepared)?;
            pipeline_bail_if_cancelled("effects")?;
            let checked = complete_checks(analysis, SemanticContext::Isolated)
                .map_err(lift_semantic_rejection)?;
            pipeline_bail_if_cancelled("linearity")?;
            lower_checked(checked, mode).map(PipelineOutcome::Lowered)
        }
    }
}

fn require_accepted_analysis(
    prepared: PreparedProgram,
) -> Result<PreparedTypeAnalysis, PipelineRejection> {
    pipeline_bail_if_cancelled("check")?;
    let outcome = analyze_prepared(prepared);
    pipeline_bail_if_cancelled("check")?;
    match outcome {
        PreparedTypeAnalysisOutcome::Accepted(analysis) => Ok(*analysis),
        PreparedTypeAnalysisOutcome::Rejected { fitness } => {
            Err(PipelineRejection::Type { fitness })
        }
    }
}

/// Parse and expand source into canonical prepared Deep.
pub fn prepare_source(
    source_kind: SourceKind,
    source: &str,
    entry: Option<&str>,
) -> Result<PreparedProgram, PreparationError> {
    match source_kind {
        SourceKind::Surf => {
            let decls = chelis_surf::parser::parse_str(source).map_err(|error| {
                PreparationError::SurfParse {
                    source: source.to_string(),
                    error,
                }
            })?;
            prepare_surf_decls(&decls, entry)
        }
        SourceKind::Deep => {
            // chelis#1088: the checked/compiled/evaluated Deep path shares the
            // stamped `.dp` ingress with the CLI. A top-level form that is
            // neither a `(module ...)` wrapper nor a declaration is rejected
            // here rather than reaching the checker as an untyped carrier.
            let exprs =
                chelis_deep::parse_and_stamp_file(source).map_err(PreparationError::DeepParse)?;
            Ok(prepare_deep(exprs, entry))
        }
    }
}

/// Expand parsed Surf declarations into prepared Deep.
pub fn prepare_surf_decls(
    decls: &[chelis_surf::ast::Decl],
    entry: Option<&str>,
) -> Result<PreparedProgram, PreparationError> {
    let desugared = chelis_surf::desugar::desugar_program(decls);
    let expanded =
        chelis_macros::expand_program(&desugared, &chelis_macros::ExpansionOptions::default())
            .map_err(PreparationError::Expansion)?
            .into_exprs();
    Ok(prepare_deep(expanded, entry))
}

/// Wrap already-expanded Deep and apply optional entry pruning.
pub fn prepare_deep(exprs: Vec<DeepExpr>, entry: Option<&str>) -> PreparedProgram {
    let expanded_deep = match entry {
        Some(entry) => crate::prune::prune_to_entry(exprs, entry),
        None => exprs,
    };
    PreparedProgram::from_expanded_deep(expanded_deep)
}

/// Lower a checked library carrier without target-specific emission.
pub fn lower_library(library: &CheckedLibrary) -> Result<LoweredLibrary, PipelineRejection> {
    pipeline_bail_if_cancelled("lower")?;
    let result = chelis_pipeline_core::lower_library(library);
    pipeline_bail_if_cancelled("lower")?;
    result.map_err(PipelineRejection::from)
}

/// Lower an isolated checked compilation.
pub fn lower_checked(
    checked: CheckedCompilation,
    mode: LoweringMode,
) -> Result<LoweredCompilation, PipelineRejection> {
    pipeline_bail_if_cancelled("lower")?;
    let result = chelis_pipeline_core::lower_checked(checked, mode);
    pipeline_bail_if_cancelled("lower")?;
    result.map_err(PipelineRejection::from)
}

/// C host selection preserves fixed-control plans from the actual lowering;
/// ordinary sources retain the core's value-root validation.
pub fn lower_checked_for_c_execution(
    checked: CheckedCompilation,
    manifest: &chelis_types::manifest::RootManifest,
    mode: LoweringMode,
) -> Result<
    (
        LoweredCompilation,
        Option<chelis_ir::host::HostExecutionPlan>,
    ),
    PipelineRejection,
> {
    pipeline_bail_if_cancelled("lower")?;
    let result = chelis_pipeline_core::lower_checked_for_c_execution(checked, manifest, mode);
    pipeline_bail_if_cancelled("lower")?;
    result.map_err(PipelineRejection::from)
}

/// Lower a context-checked compilation against its library DAG.
pub fn lower_checked_with_context(
    checked: ContextCheckedCompilation<'_>,
    library: &LoweredLibrary,
    mode: LoweringMode,
) -> Result<LoweredCompilation, PipelineRejection> {
    pipeline_bail_if_cancelled("lower")?;
    let result = chelis_pipeline_core::lower_checked_with_context(checked, library, mode);
    pipeline_bail_if_cancelled("lower")?;
    result.map_err(PipelineRejection::from)
}

fn pipeline_bail_if_cancelled(stage: &'static str) -> Result<(), PipelineRejection> {
    if chelis_types::cancellation_requested() {
        Err(PipelineRejection::Cancelled { stage })
    } else {
        Ok(())
    }
}

fn lift_semantic_rejection(rejection: SemanticRejection) -> PipelineRejection {
    if chelis_types::cancellation_requested() {
        let stage = match rejection {
            SemanticRejection::Effects { .. } => "effects",
            SemanticRejection::Linearity { .. } => "linearity",
        };
        PipelineRejection::Cancelled { stage }
    } else {
        rejection.into()
    }
}
