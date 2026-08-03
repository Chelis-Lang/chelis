//! Canonical compiler front-end orchestration.
//!
//! This module owns production stage order. Lower crates still own each stage.
//! Consumers select a closed goal and keep presentation and backend policy outside.
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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use chelis_deep::{DeepTag, Expr as DeepExpr};
use chelis_ir::dag::{Dag, NodeId, RiscOp};
use chelis_ir::lower::{LowerDiagnostic, LoweredLibrary, top_level_lowering_map};
use chelis_types::infer::SignatureInferenceMetadata;
use chelis_types::{
    CheckedProgram, FitnessReport, InferResult, TypeAnalysisOutcome, TypeEnv, analyze_ir_program,
    clean_fitness_from_stats,
};

use crate::schema::SourceKind;

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

/// A closed policy for nonfatal lowering failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoweringMode {
    /// Surface every lowering diagnostic.
    Strict,
    /// Permit a nonfatal diagnostic when no tensor root needs the DAG.
    AllowHostOnly,
    /// Permit a nonfatal diagnostic after the CLI selects its host backend.
    AllowHostBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootBindingMode {
    Exact,
    SelectedHostBackend,
    AcceptedNonfatalRejection,
}

/// A source request for the canonical pipeline.
#[derive(Debug, Clone, Copy)]
pub struct PipelineRequest<'a> {
    pub source_kind: SourceKind,
    pub source: &'a str,
    pub entry: Option<&'a str>,
    pub goal: PipelineGoal,
}

/// A prepared Deep program. Construction owns parse, desugar, expansion, and pruning.
#[derive(Debug, Clone)]
pub struct PreparedProgram {
    expanded_deep: Vec<DeepExpr>,
}

impl PreparedProgram {
    pub fn expanded_deep(&self) -> &[DeepExpr] {
        &self.expanded_deep
    }

    pub fn into_expanded_deep(self) -> Vec<DeepExpr> {
        self.expanded_deep
    }
}

/// Typed preparation failures. Formatting stays at each consumer boundary.
#[derive(Debug)]
pub enum PreparationError {
    SurfParse {
        source: String,
        error: chelis_surf::parser::ParseError,
    },
    DeepParse(chelis_deep::parser::ParseError),
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

/// The root set whose count failed a lowering invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootCountContext {
    Program,
    NewCode,
}

/// A rejection from the semantic suffix after type analysis accepts.
#[derive(Debug)]
pub enum SemanticRejection {
    Effects {
        errors: Vec<chelis_effects::EffectError>,
    },
    Linearity {
        errors: Vec<chelis_types::errors::CheckError>,
    },
}

impl fmt::Display for SemanticRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Effects { errors } => {
                write_joined_messages(formatter, errors.iter().map(|error| error.message.as_str()))
            }
            Self::Linearity { errors } => {
                write_joined_messages(formatter, errors.iter().map(|error| error.message.as_str()))
            }
        }
    }
}

impl std::error::Error for SemanticRejection {}

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

/// A canonical name in the compiler IR namespace.
///
/// This type records the namespace. It does not restrict internal aliases or dotted names.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IrName(String);

impl IrName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for IrName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl From<String> for IrName {
    fn from(name: String) -> Self {
        Self(name)
    }
}

impl From<&str> for IrName {
    fn from(name: &str) -> Self {
        Self(name.to_string())
    }
}

/// All canonical output names. This collection also contains host outputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllRootNames(Vec<IrName>);

impl AllRootNames {
    pub fn as_slice(&self) -> &[IrName] {
        &self.0
    }

    pub fn iter(&self) -> std::slice::Iter<'_, IrName> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn contains(&self, name: &IrName) -> bool {
        self.0.contains(name)
    }

    pub fn into_names(self) -> Vec<IrName> {
        self.0
    }
}

/// Canonical names for outputs that lower to DAG roots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRootNames(Vec<IrName>);

impl TensorRootNames {
    /// Drop the names the lowerer recorded as contributing no root
    /// (chelis#1095).
    ///
    /// The declared set is built before lowering, from each def's declared
    /// type. One shape does not survive lowering: a `grad` whose adjoint is
    /// absent lowers to an empty value, so its def owns no root. Only the
    /// lowerer can know that, so it reports the names and this subtracts
    /// them. Every other kind of drift still reaches
    /// [`NamedRoots::aligned`] — this removes names that were positively
    /// accounted for, not names that merely failed to appear.
    fn without(&self, rootless: &BTreeSet<String>) -> Self {
        if rootless.is_empty() {
            return self.clone();
        }
        Self(
            self.0
                .iter()
                .filter(|name| !rootless.contains(name.as_str()))
                .cloned()
                .collect(),
        )
    }

    pub fn as_slice(&self) -> &[IrName] {
        &self.0
    }

    pub fn iter(&self) -> std::slice::Iter<'_, IrName> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn into_names(self) -> Vec<IrName> {
        self.0
    }
}

/// Declared tensor outputs aligned with DAG roots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedRoots(BTreeMap<IrName, NodeId>);

impl NamedRoots {
    fn aligned(
        names: &TensorRootNames,
        roots: &[NodeId],
        context: RootCountContext,
    ) -> Result<Self, PipelineRejection> {
        let expected = names.len();
        let actual = roots.len();
        if expected != actual {
            return Err(PipelineRejection::RootCount {
                context,
                expected,
                actual,
            });
        }

        Ok(Self(
            names.iter().cloned().zip(roots.iter().copied()).collect(),
        ))
    }

    fn empty() -> Self {
        Self(BTreeMap::new())
    }

    pub fn get(&self, name: &IrName) -> Option<&NodeId> {
        self.0.get(name)
    }

    pub fn iter(&self) -> std::collections::btree_map::Iter<'_, IrName, NodeId> {
        self.0.iter()
    }

    pub fn keys(&self) -> std::collections::btree_map::Keys<'_, IrName, NodeId> {
        self.0.keys()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn into_entries(self) -> std::collections::btree_map::IntoIter<IrName, NodeId> {
        self.0.into_iter()
    }
}

/// Declared roots plus internal load aliases for forward graph lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardNodeIndex(BTreeMap<IrName, NodeId>);

impl ForwardNodeIndex {
    fn from_named_roots(named_roots: &NamedRoots, dag: &Dag) -> Self {
        let mut nodes = named_roots.0.clone();
        for node in dag.nodes() {
            if let RiscOp::Load { name } = &node.op {
                nodes.entry(IrName::new(name.as_str())).or_insert(node.id);
            }
        }
        Self(nodes)
    }

    pub fn get(&self, name: &IrName) -> Option<&NodeId> {
        self.0.get(name)
    }

    pub fn iter(&self) -> std::collections::btree_map::Iter<'_, IrName, NodeId> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn into_entries(self) -> std::collections::btree_map::IntoIter<IrName, NodeId> {
        self.0.into_iter()
    }
}

/// Canonical root names before and after tensor-root filtering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootMetadata {
    all_names: AllRootNames,
    tensor_names: TensorRootNames,
}

impl RootMetadata {
    pub fn all_names(&self) -> &AllRootNames {
        &self.all_names
    }

    pub fn tensor_names(&self) -> &TensorRootNames {
        &self.tensor_names
    }
}

/// An accepted type-analysis product with its prepared Deep program.
#[derive(Debug)]
pub struct PreparedTypeAnalysis {
    prepared: PreparedProgram,
    fitness: FitnessReport,
    program: CheckedProgram,
}

impl PreparedTypeAnalysis {
    pub fn prepared(&self) -> &PreparedProgram {
        &self.prepared
    }

    pub fn fitness(&self) -> &FitnessReport {
        &self.fitness
    }

    pub fn program(&self) -> &CheckedProgram {
        &self.program
    }
}

/// The closed result of type analysis over a prepared program.
#[derive(Debug)]
pub enum PreparedTypeAnalysisOutcome {
    Rejected { fitness: FitnessReport },
    Accepted(Box<PreparedTypeAnalysis>),
}

/// A program that passed type, effect, and linearity checks.
#[derive(Debug, Clone)]
pub struct CheckedCompilation {
    expanded_deep: Vec<DeepExpr>,
    fitness: FitnessReport,
    program: CheckedProgram,
    root_metadata: RootMetadata,
}

impl CheckedCompilation {
    pub fn expanded_deep(&self) -> &[DeepExpr] {
        &self.expanded_deep
    }

    pub fn fitness(&self) -> &FitnessReport {
        &self.fitness
    }

    pub fn program(&self) -> &CheckedProgram {
        &self.program
    }

    pub fn root_metadata(&self) -> &RootMetadata {
        &self.root_metadata
    }

    pub fn into_parts(self) -> (Vec<DeepExpr>, FitnessReport, CheckedProgram, RootMetadata) {
        (
            self.expanded_deep,
            self.fitness,
            self.program,
            self.root_metadata,
        )
    }
}

/// The named products owned by a lowered compilation.
#[derive(Debug)]
#[non_exhaustive]
pub struct LoweredParts {
    pub checked: CheckedCompilation,
    pub dag: Dag,
    pub named_roots: NamedRoots,
    pub forward_node_index: ForwardNodeIndex,
}

/// A checked program and its lowered DAG products.
#[derive(Debug)]
pub struct LoweredCompilation {
    checked: CheckedCompilation,
    dag: Dag,
    named_roots: NamedRoots,
    forward_node_index: ForwardNodeIndex,
}

impl LoweredCompilation {
    pub fn checked(&self) -> &CheckedCompilation {
        &self.checked
    }

    pub fn dag(&self) -> &Dag {
        &self.dag
    }

    pub fn into_dag(self) -> Dag {
        self.dag
    }

    pub fn named_roots(&self) -> &NamedRoots {
        &self.named_roots
    }

    pub fn forward_node_index(&self) -> &ForwardNodeIndex {
        &self.forward_node_index
    }

    pub fn into_parts(self) -> LoweredParts {
        LoweredParts {
            checked: self.checked,
            dag: self.dag,
            named_roots: self.named_roots,
            forward_node_index: self.forward_node_index,
        }
    }
}

/// A successful result for the requested goal.
#[derive(Debug)]
pub enum PipelineOutcome {
    TypeAnalysis(TypeAnalysisOutcome),
    Checked(CheckedCompilation),
    Lowered(LoweredCompilation),
}

/// Select isolated checks or checks against an accepted library.
#[derive(Debug, Clone, Copy)]
pub enum SemanticContext<'a> {
    Isolated,
    Library(&'a CheckedProgram),
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
                    let analysis = *analysis;
                    TypeAnalysisOutcome::Accepted {
                        fitness: analysis.fitness,
                        program: Box::new(analysis.program),
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
            let exprs =
                chelis_deep::parser::parse_str(source).map_err(PreparationError::DeepParse)?;
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
    PreparedProgram { expanded_deep }
}

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

/// Compose a checked library and a checked extension for internal layered paths.
pub(crate) fn compose_checked(
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

/// Lower a checked library carrier without target-specific emission.
pub fn lower_library(program: &CheckedProgram) -> Result<LoweredLibrary, PipelineRejection> {
    pipeline_bail_if_cancelled("lower")?;
    let lowered = chelis_ir::lower::try_lower_program_to_library(program);
    pipeline_bail_if_cancelled("lower")?;
    lowered.map_err(PipelineRejection::Lower)
}

/// Lower an isolated checked compilation.
pub fn lower_checked(
    checked: CheckedCompilation,
    mode: LoweringMode,
) -> Result<LoweredCompilation, PipelineRejection> {
    pipeline_bail_if_cancelled("lower")?;
    let lower_result = chelis_ir::lower::try_lower_program_to_library(checked.program());
    pipeline_bail_if_cancelled("lower")?;
    finish_isolated_lowering(checked, mode, lower_result)
}

fn finish_isolated_lowering(
    checked: CheckedCompilation,
    mode: LoweringMode,
    lower_result: Result<LoweredLibrary, LowerDiagnostic>,
) -> Result<LoweredCompilation, PipelineRejection> {
    pipeline_bail_if_cancelled("lower")?;
    let (dag, rootless_defs, root_binding_mode) = match lower_result {
        Ok(library) if mode == LoweringMode::AllowHostBackend && library.dag.roots().is_empty() => {
            (
                library.dag,
                library.rootless_defs,
                RootBindingMode::SelectedHostBackend,
            )
        }
        Ok(library) => (library.dag, library.rootless_defs, RootBindingMode::Exact),
        Err(diagnostic)
            if mode == LoweringMode::AllowHostOnly
                && !diagnostic.fatal
                && checked.root_metadata.tensor_names.is_empty() =>
        {
            (
                Dag::new(),
                BTreeSet::new(),
                RootBindingMode::AcceptedNonfatalRejection,
            )
        }
        Err(diagnostic) if mode == LoweringMode::AllowHostBackend && !diagnostic.fatal => (
            Dag::new(),
            BTreeSet::new(),
            RootBindingMode::AcceptedNonfatalRejection,
        ),
        Err(diagnostic) => return Err(PipelineRejection::Lower(diagnostic)),
    };
    finish_lowering(
        checked,
        dag,
        &rootless_defs,
        RootCountContext::Program,
        root_binding_mode,
    )
}

/// Lower a checked compilation against a reusable library DAG.
pub fn lower_checked_with_context(
    mut checked: CheckedCompilation,
    library: &LoweredLibrary,
    mode: LoweringMode,
) -> Result<LoweredCompilation, PipelineRejection> {
    pipeline_bail_if_cancelled("lower")?;
    let lowered_map = chelis_ir::lower::top_level_lowering_map_with_context(
        library,
        checked.program.exprs(),
        checked.program.type_env(),
    );
    checked.root_metadata = root_metadata(&checked.program, Some(&lowered_map));
    let tensor_names = checked.root_metadata.tensor_names.clone();

    let lower_result = chelis_ir::lower::try_lower_program_with_context(library, &checked.program);
    pipeline_bail_if_cancelled("lower")?;
    let (mut dag, rootless_defs, accepted_nonfatal_rejection) = match lower_result {
        Ok(composed) => (composed.dag, composed.rootless_defs, false),
        Err(diagnostic)
            if mode == LoweringMode::AllowHostOnly
                && !diagnostic.fatal
                && tensor_names.is_empty() =>
        {
            (library.dag.clone(), BTreeSet::new(), true)
        }
        Err(diagnostic) => return Err(PipelineRejection::Lower(diagnostic)),
    };

    let library_root_count = library.dag.roots().len();
    let root_start = library_root_count.min(dag.roots().len());
    let new_roots = dag.roots()[root_start..].to_vec();
    dag.set_roots(new_roots);
    let root_binding_mode = if accepted_nonfatal_rejection {
        RootBindingMode::AcceptedNonfatalRejection
    } else if mode == LoweringMode::AllowHostBackend && dag.roots().is_empty() {
        RootBindingMode::SelectedHostBackend
    } else {
        RootBindingMode::Exact
    };
    finish_lowering(
        checked,
        dag,
        &rootless_defs,
        RootCountContext::NewCode,
        root_binding_mode,
    )
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

fn finish_lowering(
    checked: CheckedCompilation,
    dag: Dag,
    rootless_defs: &BTreeSet<String>,
    root_context: RootCountContext,
    root_binding_mode: RootBindingMode,
) -> Result<LoweredCompilation, PipelineRejection> {
    let named_roots = match root_binding_mode {
        RootBindingMode::Exact => NamedRoots::aligned(
            &checked.root_metadata.tensor_names.without(rootless_defs),
            dag.roots(),
            root_context,
        )?,
        RootBindingMode::SelectedHostBackend | RootBindingMode::AcceptedNonfatalRejection => {
            NamedRoots::empty()
        }
    };
    let forward_node_index = ForwardNodeIndex::from_named_roots(&named_roots, &dag);

    Ok(LoweredCompilation {
        checked,
        dag,
        named_roots,
        forward_node_index,
    })
}

fn root_metadata(
    program: &CheckedProgram,
    lowered_names: Option<&HashMap<String, bool>>,
) -> RootMetadata {
    let all_names = AllRootNames(root_names_from_checked_exprs(
        program.exprs(),
        program.type_env(),
        None,
    ));
    let owned_lowered_names;
    let lowered_names = match lowered_names {
        Some(map) => Some(map),
        None => {
            owned_lowered_names = top_level_lowering_map(program.exprs(), program.type_env());
            Some(&owned_lowered_names)
        }
    };
    let tensor_names = TensorRootNames(root_names_from_checked_exprs(
        program.exprs(),
        program.type_env(),
        lowered_names,
    ));
    RootMetadata {
        all_names,
        tensor_names,
    }
}

fn root_names_from_checked_exprs(
    exprs: &[DeepExpr],
    type_env: &HashMap<String, DeepExpr>,
    lowered_names: Option<&HashMap<String, bool>>,
) -> Vec<IrName> {
    let mut names = Vec::new();
    for expr in exprs {
        collect_checked_decl_names(expr, type_env, lowered_names, &mut names);
    }
    names
}

fn collect_checked_decl_names(
    expr: &DeepExpr,
    type_env: &HashMap<String, DeepExpr>,
    lowered_names: Option<&HashMap<String, bool>>,
    output: &mut Vec<IrName>,
) {
    let Some((tag, children)) = tagged_children(expr) else {
        return;
    };
    match tag {
        DeepTag::Module => {
            for child in children.iter().skip(1) {
                collect_checked_decl_names(child, type_env, lowered_names, output);
            }
        }
        DeepTag::Def => {
            if let Some(name) = children.first().and_then(symbol_name) {
                if lowered_names.is_some_and(|map| !map.get(name).copied().unwrap_or(false)) {
                    return;
                }
                let value = children.get(1);
                let ty = type_env
                    .get(name)
                    .or_else(|| value.and_then(expr_type_metadata));
                extend_root_names(name, ty, value, output);
            }
        }
        _ => {}
    }
}

fn tagged_children(expr: &DeepExpr) -> Option<(DeepTag, &[DeepExpr])> {
    match expr {
        DeepExpr::List(list, _) => Some((list.tag()?, list.elements.get(2..)?)),
        DeepExpr::Node(node, _) => Some((node.tag(), node.children_slice())),
        _ => None,
    }
}

fn extend_root_names(
    name: &str,
    ty: Option<&DeepExpr>,
    value: Option<&DeepExpr>,
    output: &mut Vec<IrName>,
) {
    if let Some((tag, children)) = ty.and_then(tagged_children) {
        if tag == DeepTag::TFn {
            extend_root_names(name, children.last(), None, output);
            return;
        }
        if tag == DeepTag::TTuple {
            for (index, child) in children.iter().enumerate() {
                extend_root_names(&format!("{name}.{index}"), Some(child), None, output);
            }
            return;
        }
    }
    if let Some((DeepTag::Tuple, children)) = value.and_then(tagged_children) {
        for (index, child) in children.iter().enumerate() {
            extend_root_names(
                &format!("{name}.{index}"),
                expr_type_metadata(child),
                Some(child),
                output,
            );
        }
        return;
    }
    output.push(IrName::new(name));
}

fn expr_type_metadata(expr: &DeepExpr) -> Option<&DeepExpr> {
    let metadata = match expr {
        DeepExpr::List(list, _) => match list.elements.get(1) {
            Some(DeepExpr::Map(metadata, _)) => metadata,
            _ => return None,
        },
        DeepExpr::Node(node, _) => node.meta(),
        _ => return None,
    };
    metadata
        .entries
        .iter()
        .find(|(key, _)| key == "type")
        .map(|(_, value)| value)
}

fn symbol_name(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::Atom(chelis_deep::Atom::Name(name), _) => Some(name.as_str()),
        _ => None,
    }
}

#[cfg(test)]
mod artifact_type_tests {
    use super::*;

    fn checked_compilation(source: &str) -> CheckedCompilation {
        let prepared =
            prepare_source(SourceKind::Surf, source, None).expect("test source must prepare");
        let analysis = match analyze_prepared(prepared) {
            PreparedTypeAnalysisOutcome::Accepted(analysis) => *analysis,
            PreparedTypeAnalysisOutcome::Rejected { fitness } => {
                panic!("test source must type-check: {:?}", fitness.errors)
            }
        };
        complete_checks(analysis, SemanticContext::Isolated)
            .expect("test source must pass semantic checks")
    }

    /// A lowering result carrying `dag` and nothing else. chelis#1095 made
    /// `finish_isolated_lowering` consume the carrier rather than a bare
    /// `Dag`, so it can read the reported rootless defs.
    fn lowered(dag: Dag) -> LoweredLibrary {
        LoweredLibrary {
            dag,
            symbol_table: HashMap::new(),
            program_defs: HashMap::new(),
            program_types: HashMap::new(),
            linearity: Default::default(),
            lowered_names: HashMap::new(),
            rootless_defs: BTreeSet::new(),
        }
    }

    #[test]
    fn strict_successful_empty_dag_rejects_nonempty_tensor_root_names() {
        let checked =
            checked_compilation("def identity(x: tensor[n, f32]) -> tensor[n, f32] = x\n");

        let error =
            finish_isolated_lowering(checked, LoweringMode::Strict, Ok(lowered(Dag::new())))
                .expect_err("strict successful lowering must use exact root alignment");

        assert!(matches!(
            error,
            PipelineRejection::RootCount {
                context: RootCountContext::Program,
                expected: 1,
                actual: 0,
            }
        ));
    }

    #[test]
    fn successful_empty_dag_aligns_empty_tensor_root_names() {
        let checked = checked_compilation("label = \"host only\"\n");

        let lowered =
            finish_isolated_lowering(checked, LoweringMode::Strict, Ok(lowered(Dag::new())))
                .expect("empty names and empty roots must align");

        assert!(lowered.dag().roots().is_empty());
        assert!(lowered.named_roots().is_empty());
    }

    #[test]
    fn selected_host_backend_accepts_a_successful_empty_dag() {
        let checked =
            checked_compilation("def identity(x: tensor[n, f32]) -> tensor[n, f32] = x\n");

        let lowered = finish_isolated_lowering(
            checked,
            LoweringMode::AllowHostBackend,
            Ok(lowered(Dag::new())),
        )
        .expect("the selected host backend owns the output");

        assert!(lowered.dag().roots().is_empty());
        assert!(lowered.named_roots().is_empty());
    }

    #[test]
    fn selected_host_backend_accepts_a_nonfatal_lower_rejection() {
        let checked =
            checked_compilation("def identity(x: tensor[n, f32]) -> tensor[n, f32] = x\n");
        let diagnostic = LowerDiagnostic {
            message: "host backend required".to_string(),
            span: None,
            span_id: None,
            fatal: false,
        };

        let lowered =
            finish_isolated_lowering(checked, LoweringMode::AllowHostBackend, Err(diagnostic))
                .expect("the selected host backend accepts a nonfatal rejection");

        assert!(lowered.dag().roots().is_empty());
        assert!(lowered.named_roots().is_empty());
    }

    #[test]
    fn symbol_name_reads_the_typed_deep_name_atom() {
        let expression = DeepExpr::Atom(
            chelis_deep::Atom::Name("root".to_string()),
            chelis_deep::Span::new(0, 4),
        );

        assert_eq!(symbol_name(&expression), Some("root"));
    }

    #[test]
    fn typed_node_root_collection_matches_list_root_collection() {
        let source = "(module {} m (def {} out (tuple {} \
            (lit {type: (t-prim {} f32)} 1.0) \
            (lit {type: (t-prim {} f32)} 2.0))))";
        let list_exprs = chelis_deep::parser::parse_str(source).expect("list Deep must parse");
        let typed_exprs =
            chelis_deep::parse_and_stamp_file(source).expect("typed Deep must parse and stamp");
        assert!(matches!(typed_exprs.first(), Some(DeepExpr::Node(_, _))));

        let expected = vec![IrName::new("out.0"), IrName::new("out.1")];
        assert_eq!(
            root_names_from_checked_exprs(&list_exprs, &HashMap::new(), None),
            expected
        );
        assert_eq!(
            root_names_from_checked_exprs(&typed_exprs, &HashMap::new(), None),
            expected
        );
    }

    #[test]
    fn typed_node_non_root_declaration_does_not_create_a_root_name() {
        let source = "(module {} m (defsig {} f (t-fn {} (t-prim {} f32))))";
        let typed_exprs =
            chelis_deep::parse_and_stamp_file(source).expect("typed Deep must parse and stamp");

        assert!(root_names_from_checked_exprs(&typed_exprs, &HashMap::new(), None).is_empty());
    }

    // chelis#1095: the lowerer reports which defs held no tensor node, and
    // those names come out of the declared set before alignment.
    #[test]
    fn declared_root_names_drop_the_names_the_lowerer_reports_as_rootless() {
        let names = TensorRootNames(vec![
            IrName::new("sumsq"),
            IrName::new("grad_sumsq"),
            IrName::new("ho_ignores"),
        ]);
        let rootless = BTreeSet::from(["grad_sumsq".to_string()]);

        let kept = names.without(&rootless);
        let kept: Vec<&str> = kept.iter().map(IrName::as_str).collect();
        assert_eq!(kept, ["sumsq", "ho_ignores"]);
    }

    // Negative parity, and the rt-1103 regression in unit form: a def is
    // dropped ONLY when the lowerer reported it. An empty report must leave
    // every declared name in place — the over-classification that deleted
    // `ho_a`'s kernel showed up exactly here, as names disappearing that no
    // lowering outcome justified.
    #[test]
    fn declared_root_names_are_untouched_when_nothing_is_reported_rootless() {
        let names = TensorRootNames(vec![IrName::new("ho_a"), IrName::new("sumsq")]);

        let kept = names.without(&BTreeSet::new());
        let kept: Vec<&str> = kept.iter().map(IrName::as_str).collect();
        assert_eq!(kept, ["ho_a", "sumsq"]);
    }

    // A rootless def is subtracted from the DECLARED roots only. The full
    // inventory is a different product and keeps every declaration.
    #[test]
    fn a_rootless_grad_def_leaves_the_full_name_inventory_alone() {
        let checked = checked_compilation(
            "def sumsq(theta: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(theta, theta), 0))\n\
             def grad_sumsq(model: tensor[3, f32] -> f32, theta: tensor[3, f32]) -> tensor[3, f32] = {\n\
             \x20 target = fn (theta_local: tensor[3, f32]) -> model(theta_local)\n\
             \x20 grad(target, wrt=theta_local)(theta)\n\
             }\n",
        );

        let all_names: Vec<&str> = checked
            .root_metadata()
            .all_names()
            .iter()
            .map(IrName::as_str)
            .collect();
        assert_eq!(all_names, ["sumsq", "grad_sumsq"]);

        // Both are still DECLARED roots pre-lowering — the subtraction is a
        // lowering outcome, not a signature judgement.
        let declared: Vec<&str> = checked
            .root_metadata()
            .tensor_names()
            .iter()
            .map(IrName::as_str)
            .collect();
        assert_eq!(declared, ["sumsq", "grad_sumsq"]);
    }

    #[test]
    fn named_roots_reject_a_count_mismatch_without_a_partial_map() {
        let names = TensorRootNames(vec![IrName::new("first"), IrName::new("second")]);
        let error = NamedRoots::aligned(&names, &[NodeId(7)], RootCountContext::Program)
            .expect_err("different counts must reject before map construction");

        assert!(matches!(
            error,
            PipelineRejection::RootCount {
                context: RootCountContext::Program,
                expected: 2,
                actual: 1,
            }
        ));
    }
}
