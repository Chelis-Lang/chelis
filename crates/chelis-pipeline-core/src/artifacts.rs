use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use chelis_deep::Expr as DeepExpr;
use chelis_ir::Dag;
use chelis_ir::dag::NodeId;
use chelis_ir::lower::LowerDiagnostic;
use chelis_types::{CheckedProgram, FitnessReport, InferResult, TypeEnv};

/// A closed policy for nonfatal lowering failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoweringMode {
    /// Surface every lowering diagnostic.
    Strict,
    /// Permit a nonfatal diagnostic when no tensor root needs the DAG.
    AllowHostOnly,
    /// Permit a nonfatal diagnostic after the facade selects its host backend.
    AllowHostBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RootBindingMode {
    Exact,
    SelectedHostBackend,
    AcceptedNonfatalRejection,
}

/// An owned expanded Deep program at the semantic-core boundary.
#[derive(Debug, Clone)]
pub struct PreparedProgram {
    pub(crate) expanded_deep: Vec<DeepExpr>,
}

impl PreparedProgram {
    /// Construct the core carrier after an upper layer completes expansion.
    pub fn from_expanded_deep(expanded_deep: Vec<DeepExpr>) -> Self {
        Self { expanded_deep }
    }

    pub fn expanded_deep(&self) -> &[DeepExpr] {
        &self.expanded_deep
    }

    pub fn into_expanded_deep(self) -> Vec<DeepExpr> {
        self.expanded_deep
    }
}

/// The root set whose count failed a lowering invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootCountContext {
    Program,
    NewCode,
}

/// The public context for standalone semantic completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticContext {
    Isolated,
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

/// A rejection from checked-library construction or cache parsing.
#[derive(Debug)]
pub enum LibraryRejection {
    Type {
        report: InferResult,
    },
    ContextMismatch,
    Effects {
        errors: Vec<chelis_effects::EffectError>,
    },
    Linearity {
        errors: Vec<chelis_types::errors::CheckError>,
    },
}

impl fmt::Display for LibraryRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Type { report } => write_joined_messages(
                formatter,
                report.errors.iter().map(|error| error.message.as_str()),
            ),
            Self::ContextMismatch => {
                formatter.write_str("the type environment does not match the checked library")
            }
            Self::Effects { errors } => {
                write_joined_messages(formatter, errors.iter().map(|error| error.message.as_str()))
            }
            Self::Linearity { errors } => {
                write_joined_messages(formatter, errors.iter().map(|error| error.message.as_str()))
            }
        }
    }
}

impl std::error::Error for LibraryRejection {}

/// A lower-layer failure from core lowering and exact root construction.
#[derive(Debug)]
pub enum CoreLowerError {
    Lower(LowerDiagnostic),
    RootCount {
        context: RootCountContext,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for CoreLowerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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

impl std::error::Error for CoreLowerError {}

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
pub struct AllRootNames(pub(crate) Vec<IrName>);

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
pub struct TensorRootNames(pub(crate) Vec<IrName>);

impl TensorRootNames {
    /// Drop names that the lowerer records as rootless.
    pub(crate) fn without(&self, rootless: &BTreeSet<String>) -> Self {
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
pub struct NamedRoots(pub(crate) BTreeMap<IrName, NodeId>);

impl NamedRoots {
    pub(crate) fn aligned(
        names: &TensorRootNames,
        roots: &[NodeId],
        context: RootCountContext,
    ) -> Result<Self, CoreLowerError> {
        let expected = names.len();
        let actual = roots.len();
        if expected != actual {
            return Err(CoreLowerError::RootCount {
                context,
                expected,
                actual,
            });
        }

        Ok(Self(
            names.iter().cloned().zip(roots.iter().copied()).collect(),
        ))
    }

    pub(crate) fn empty() -> Self {
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
pub struct ForwardNodeIndex(pub(crate) BTreeMap<IrName, NodeId>);

impl ForwardNodeIndex {
    pub(crate) fn from_named_roots(named_roots: &NamedRoots, dag: &Dag) -> Self {
        let mut nodes = named_roots.0.clone();
        for node in dag.nodes() {
            if let chelis_ir::dag::RiscOp::Load { name } = &node.op {
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
    pub(crate) all_names: AllRootNames,
    pub(crate) tensor_names: TensorRootNames,
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
    pub(crate) prepared: PreparedProgram,
    pub(crate) fitness: FitnessReport,
    pub(crate) program: CheckedProgram,
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

    pub fn into_parts(self) -> (PreparedProgram, FitnessReport, CheckedProgram) {
        (self.prepared, self.fitness, self.program)
    }
}

/// Library type analysis with both products from one inference session.
#[derive(Debug)]
pub struct PreparedLibraryAnalysis {
    pub(crate) type_env: TypeEnv,
    pub(crate) analysis: PreparedTypeAnalysis,
}

impl PreparedLibraryAnalysis {
    pub fn type_env(&self) -> &TypeEnv {
        &self.type_env
    }

    pub fn analysis(&self) -> &PreparedTypeAnalysis {
        &self.analysis
    }
}

/// The closed result of type analysis over a prepared program.
#[derive(Debug)]
pub enum PreparedTypeAnalysisOutcome {
    Rejected { fitness: FitnessReport },
    Accepted(Box<PreparedTypeAnalysis>),
}

/// A type environment bound to one semantically accepted library program.
#[derive(Debug, Clone)]
pub struct CheckedLibrary {
    pub(crate) type_env: TypeEnv,
    pub(crate) program: CheckedProgram,
}

impl CheckedLibrary {
    pub fn type_env(&self) -> &TypeEnv {
        &self.type_env
    }

    pub fn program(&self) -> &CheckedProgram {
        &self.program
    }
}

/// Contextual type analysis bound to the library that produced it.
#[derive(Debug)]
pub struct ContextualTypeAnalysis<'library> {
    pub(crate) library: &'library CheckedLibrary,
    pub(crate) analysis: PreparedTypeAnalysis,
}

impl<'library> ContextualTypeAnalysis<'library> {
    pub fn library(&self) -> &'library CheckedLibrary {
        self.library
    }

    pub fn analysis(&self) -> &PreparedTypeAnalysis {
        &self.analysis
    }
}

/// Library-extension analysis with every product from one contextual session.
#[derive(Debug)]
pub struct ContextualLibraryTypeAnalysis<'library> {
    pub(crate) library: &'library CheckedLibrary,
    pub(crate) type_env: TypeEnv,
    pub(crate) analysis: PreparedTypeAnalysis,
}

impl<'library> ContextualLibraryTypeAnalysis<'library> {
    pub fn library(&self) -> &'library CheckedLibrary {
        self.library
    }

    pub fn type_env(&self) -> &TypeEnv {
        &self.type_env
    }

    pub fn analysis(&self) -> &PreparedTypeAnalysis {
        &self.analysis
    }
}

/// Semantic success for an extension checked against one exact library.
#[derive(Debug)]
pub struct ContextCheckedCompilation<'library> {
    pub(crate) library: &'library CheckedLibrary,
    pub(crate) extension: CheckedCompilation,
}

impl<'library> ContextCheckedCompilation<'library> {
    pub fn library(&self) -> &'library CheckedLibrary {
        self.library
    }

    pub fn extension(&self) -> &CheckedCompilation {
        &self.extension
    }

    pub(crate) fn into_extension(self) -> CheckedCompilation {
        self.extension
    }
}

/// A program that passed type, effect, and linearity checks.
#[derive(Debug, Clone)]
pub struct CheckedCompilation {
    pub(crate) expanded_deep: Vec<DeepExpr>,
    pub(crate) fitness: FitnessReport,
    pub(crate) program: CheckedProgram,
    pub(crate) root_metadata: RootMetadata,
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
    pub(crate) checked: CheckedCompilation,
    pub(crate) dag: Dag,
    pub(crate) named_roots: NamedRoots,
    pub(crate) forward_node_index: ForwardNodeIndex,
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
