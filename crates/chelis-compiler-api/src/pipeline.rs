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

use std::collections::{BTreeMap, HashMap};
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

    pub fn into_names(self) -> Vec<IrName> {
        self.0
    }
}

/// Canonical names for outputs that lower to DAG roots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRootNames(Vec<IrName>);

impl TensorRootNames {
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
    let prepared = prepare_source(request.source_kind, request.source, request.entry)
        .map_err(PipelineRejection::Preparation)?;
    run_prepared(prepared, request.goal)
}

/// Run a prepared program through the required pipeline prefix.
pub fn run_prepared(
    prepared: PreparedProgram,
    goal: PipelineGoal,
) -> Result<PipelineOutcome, PipelineRejection> {
    match goal {
        PipelineGoal::TypeAnalysis => Ok(PipelineOutcome::TypeAnalysis(
            match analyze_prepared(prepared) {
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
            },
        )),
        PipelineGoal::FullCheck => {
            let analysis = require_accepted_analysis(prepared)?;
            complete_checks(analysis, SemanticContext::Isolated)
                .map(PipelineOutcome::Checked)
                .map_err(PipelineRejection::from)
        }
        PipelineGoal::Lower(mode) => {
            let analysis = require_accepted_analysis(prepared)?;
            let checked = complete_checks(analysis, SemanticContext::Isolated)?;
            lower_checked(checked, mode).map(PipelineOutcome::Lowered)
        }
    }
}

fn require_accepted_analysis(
    prepared: PreparedProgram,
) -> Result<PreparedTypeAnalysis, PipelineRejection> {
    match analyze_prepared(prepared) {
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
    chelis_ir::lower::try_lower_program_to_library(program).map_err(PipelineRejection::Lower)
}

/// Lower an isolated checked compilation.
pub fn lower_checked(
    checked: CheckedCompilation,
    mode: LoweringMode,
) -> Result<LoweredCompilation, PipelineRejection> {
    let (dag, host_backend_fallback) = match chelis_ir::lower::try_lower_program(checked.program())
    {
        Ok(dag) => {
            let host_backend_fallback =
                mode == LoweringMode::AllowHostBackend && dag.roots().is_empty();
            (dag, host_backend_fallback)
        }
        Err(diagnostic)
            if mode == LoweringMode::AllowHostOnly
                && !diagnostic.fatal
                && checked.root_metadata.tensor_names.is_empty() =>
        {
            (Dag::new(), true)
        }
        Err(diagnostic) if mode == LoweringMode::AllowHostBackend && !diagnostic.fatal => {
            (Dag::new(), true)
        }
        Err(diagnostic) => return Err(PipelineRejection::Lower(diagnostic)),
    };
    finish_lowering(
        checked,
        dag,
        RootCountContext::Program,
        host_backend_fallback,
    )
}

/// Lower a checked compilation against a reusable library DAG.
pub fn lower_checked_with_context(
    mut checked: CheckedCompilation,
    library: &LoweredLibrary,
    mode: LoweringMode,
) -> Result<LoweredCompilation, PipelineRejection> {
    let lowered_map = chelis_ir::lower::top_level_lowering_map_with_context(
        library,
        checked.program.exprs(),
        checked.program.type_env(),
    );
    checked.root_metadata = root_metadata(&checked.program, Some(&lowered_map));
    let tensor_names = checked.root_metadata.tensor_names.clone();

    let mut dag = match chelis_ir::lower::try_lower_program_with_context(library, &checked.program)
    {
        Ok(dag) => dag,
        Err(diagnostic)
            if mode == LoweringMode::AllowHostOnly
                && !diagnostic.fatal
                && tensor_names.is_empty() =>
        {
            library.dag.clone()
        }
        Err(diagnostic) => return Err(PipelineRejection::Lower(diagnostic)),
    };

    let library_root_count = library.dag.roots().len();
    let root_start = library_root_count.min(dag.roots().len());
    let new_roots = dag.roots()[root_start..].to_vec();
    dag.set_roots(new_roots);
    finish_lowering(checked, dag, RootCountContext::NewCode, false)
}

fn finish_lowering(
    checked: CheckedCompilation,
    dag: Dag,
    root_context: RootCountContext,
    allow_empty_host_fallback: bool,
) -> Result<LoweredCompilation, PipelineRejection> {
    let named_roots = if allow_empty_host_fallback && dag.roots().is_empty() {
        NamedRoots::empty()
    } else {
        NamedRoots::aligned(
            &checked.root_metadata.tensor_names,
            dag.roots(),
            root_context,
        )?
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
    let DeepExpr::List(list, _) = expr else {
        return;
    };
    match list.tag() {
        Some(DeepTag::Module) => {
            for child in list.elements.iter().skip(3) {
                collect_checked_decl_names(child, type_env, lowered_names, output);
            }
        }
        Some(DeepTag::Def) => {
            if let Some(name) = list.elements.get(2).and_then(symbol_name) {
                if lowered_names.is_some_and(|map| !map.get(name).copied().unwrap_or(false)) {
                    return;
                }
                let value = list.elements.get(3);
                let ty = type_env
                    .get(name)
                    .or_else(|| value.and_then(expr_type_metadata));
                extend_root_names(name, ty, value, output);
            }
        }
        _ => {}
    }
}

fn extend_root_names(
    name: &str,
    ty: Option<&DeepExpr>,
    value: Option<&DeepExpr>,
    output: &mut Vec<IrName>,
) {
    if let Some(DeepExpr::List(list, _)) = ty
        && let Some(tag) = list.tag()
    {
        if tag == DeepTag::TFn {
            extend_root_names(name, list.elements.last(), None, output);
            return;
        }
        if tag == DeepTag::TTuple {
            for (index, child) in list.elements.iter().skip(2).enumerate() {
                extend_root_names(&format!("{name}.{index}"), Some(child), None, output);
            }
            return;
        }
    }
    if let Some(DeepExpr::List(list, _)) = value
        && list.tag() == Some(DeepTag::Tuple)
    {
        for (index, child) in list.elements.iter().skip(2).enumerate() {
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
    let DeepExpr::List(list, _) = expr else {
        return None;
    };
    match list.elements.get(1) {
        Some(DeepExpr::Map(metadata, _)) => metadata
            .entries
            .iter()
            .find(|(key, _)| key == "type")
            .map(|(_, value)| value),
        _ => None,
    }
}

fn symbol_name(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::Atom(chelis_deep::Atom::Symbol(name), _) => Some(name.as_str()),
        _ => None,
    }
}

#[cfg(test)]
mod artifact_type_tests {
    use super::*;

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
