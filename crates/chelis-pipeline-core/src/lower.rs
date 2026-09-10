use std::collections::{BTreeMap, BTreeSet};

use chelis_ir::Dag;
use chelis_ir::lower::{LowerDiagnostic, LoweredLibrary as IrLoweredLibrary};

use crate::artifacts::RootBindingMode;
use crate::roots::root_metadata;
use crate::{
    CheckedCompilation, CheckedLibrary, ContextCheckedCompilation, CoreLowerError,
    ForwardNodeIndex, LoweredCompilation, LoweringMode, NamedRoots, RootCountContext,
    TensorRootNames,
};

/// An immutable lowered payload bound to a semantically checked library.
///
/// Only [`lower_library`] can construct this artifact. Cache adapters can read
/// its raw carrier, but they cannot replace the carrier or its proof identity.
#[derive(Debug, Clone)]
pub struct LoweredLibrary {
    inner: IrLoweredLibrary,
}

impl LoweredLibrary {
    /// Return the immutable library DAG.
    pub fn dag(&self) -> &Dag {
        self.inner.dag()
    }

    /// Return the immutable map of lowering decisions.
    pub fn lowered_names(&self) -> &BTreeMap<String, bool> {
        self.inner.lowered_names()
    }

    /// Return the checked-library identity for this lowered payload.
    pub fn library_proof_id(&self) -> Option<chelis_types::LibraryProofId> {
        self.inner.library_proof_id()
    }

    /// Return the immutable raw carrier for a cache wire adapter.
    pub fn raw(&self) -> &IrLoweredLibrary {
        &self.inner
    }
}

/// Lower a checked library carrier without target-specific emission.
pub fn lower_library(library: &CheckedLibrary) -> Result<LoweredLibrary, CoreLowerError> {
    chelis_ir::lower::try_lower_program_to_library(library.program())
        .map(|inner| LoweredLibrary { inner })
        .map_err(CoreLowerError::Lower)
}

/// Lower an isolated checked compilation.
pub fn lower_checked(
    checked: CheckedCompilation,
    mode: LoweringMode,
) -> Result<LoweredCompilation, CoreLowerError> {
    let lower_result =
        chelis_ir::lower::try_lower_program_to_library(checked.program()).map(LoweredProgram::from);
    finish_isolated_lowering(checked, mode, lower_result)
}

/// Additive evaluator products derived from the same sealed checked source.
/// Ordinary lowering/cache carriers remain unchanged and cannot stand in for
/// this non-serialized execution transport.
pub fn lower_checked_for_evaluation(
    checked: &CheckedCompilation,
) -> Result<chelis_ir::lower::EvaluationProgram, CoreLowerError> {
    chelis_ir::lower::try_lower_program_to_evaluation_library(checked.program())
        .map(|library| library.program().clone())
        .map_err(CoreLowerError::Lower)
}

pub fn lower_checked_with_evaluation_context(
    checked: &ContextCheckedCompilation<'_>,
    evaluation: &chelis_ir::lower::EvaluationLibrary,
) -> Result<chelis_ir::lower::EvaluationProgram, CoreLowerError> {
    if evaluation.library_for_inspection().library_proof_id()
        != checked.library().program().library_proof_id()
    {
        return Err(CoreLowerError::Lower(LowerDiagnostic {
            message: "the evaluation library does not match its checked context".into(),
            span: None,
            span_id: None,
            fatal: true,
        }));
    }
    chelis_ir::lower::try_lower_program_with_evaluation_context(
        evaluation,
        checked.extension().program(),
    )
    .map_err(CoreLowerError::Lower)
}

struct LoweredProgram {
    dag: Dag,
    rootless_defs: BTreeSet<String>,
}

impl From<IrLoweredLibrary> for LoweredProgram {
    fn from(library: IrLoweredLibrary) -> Self {
        let (dag, rootless_defs) = library.into_dag_and_rootless_defs();
        Self { dag, rootless_defs }
    }
}

impl From<chelis_ir::lower::ComposedLowering> for LoweredProgram {
    fn from(composed: chelis_ir::lower::ComposedLowering) -> Self {
        Self {
            dag: composed.dag,
            rootless_defs: composed.rootless_defs,
        }
    }
}

fn finish_isolated_lowering(
    checked: CheckedCompilation,
    mode: LoweringMode,
    lower_result: Result<LoweredProgram, LowerDiagnostic>,
) -> Result<LoweredCompilation, CoreLowerError> {
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
        Err(diagnostic) => return Err(CoreLowerError::Lower(diagnostic)),
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
    checked: ContextCheckedCompilation<'_>,
    library: &LoweredLibrary,
    mode: LoweringMode,
) -> Result<LoweredCompilation, CoreLowerError> {
    if library.library_proof_id() != checked.library().program().library_proof_id() {
        return Err(CoreLowerError::Lower(LowerDiagnostic {
            message: "the lowered library does not match the context-checked program".to_string(),
            span: None,
            span_id: None,
            fatal: true,
        }));
    }
    let mut checked = checked.into_extension();
    let lowered_map = chelis_ir::lower::top_level_lowering_map_with_context(
        library.raw(),
        checked.program.exprs(),
        checked.program.type_env(),
    );
    checked.root_metadata = root_metadata(&checked.program, Some(&lowered_map));
    let tensor_names = checked.root_metadata.tensor_names.clone();

    let lower_result =
        chelis_ir::lower::try_lower_program_with_context(library.raw(), &checked.program)
            .map(LoweredProgram::from);
    finish_contextual_lowering(checked, library.dag(), mode, lower_result, &tensor_names)
}

/// Bind new-code roots after contextual lowering, applying the same host
/// policy the isolated path applies. Under `AllowHostBackend` a nonfatal lower
/// rejection yields an empty host result, matching `finish_isolated_lowering`,
/// so the two paths agree for a selected host backend.
fn finish_contextual_lowering(
    checked: CheckedCompilation,
    library_dag: &Dag,
    mode: LoweringMode,
    lower_result: Result<LoweredProgram, LowerDiagnostic>,
    tensor_names: &TensorRootNames,
) -> Result<LoweredCompilation, CoreLowerError> {
    let (mut dag, rootless_defs, accepted_nonfatal_rejection) = match lower_result {
        Ok(composed) => (composed.dag, composed.rootless_defs, false),
        Err(diagnostic)
            if mode == LoweringMode::AllowHostOnly
                && !diagnostic.fatal
                && tensor_names.is_empty() =>
        {
            (library_dag.clone(), BTreeSet::new(), true)
        }
        Err(diagnostic) if mode == LoweringMode::AllowHostBackend && !diagnostic.fatal => {
            (library_dag.clone(), BTreeSet::new(), true)
        }
        Err(diagnostic) => return Err(CoreLowerError::Lower(diagnostic)),
    };

    // The composed DAG prepends the library's roots, so new-code roots begin
    // at `library_root_count`. The `.min` is a defensive slice guard: a healthy
    // composition always has at least the library's roots, so a composed DAG
    // with fewer roots than its library indicates an upstream lowering defect.
    // The clamp keeps that from panicking on the slice; the real mismatch is
    // then caught by the root-count alignment in `finish_lowering` unless the
    // new code declares no tensor names.
    let library_root_count = library_dag.roots().len();
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

fn finish_lowering(
    checked: CheckedCompilation,
    dag: Dag,
    rootless_defs: &BTreeSet<String>,
    root_context: RootCountContext,
    root_binding_mode: RootBindingMode,
) -> Result<LoweredCompilation, CoreLowerError> {
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

#[cfg(test)]
mod tests {
    use chelis_ir::dag::NodeId;

    use super::*;
    use crate::{
        IrName, PreparedProgram, PreparedTypeAnalysisOutcome, SemanticContext, TensorRootNames,
        analyze_prepared, complete_checks,
    };

    fn checked_compilation(source: &str) -> CheckedCompilation {
        let expressions = chelis_deep::parser::parse_str(source).expect("test Deep must parse");
        let prepared = PreparedProgram::from_expanded_deep(expressions);
        let analysis = match analyze_prepared(prepared) {
            PreparedTypeAnalysisOutcome::Accepted(analysis) => *analysis,
            PreparedTypeAnalysisOutcome::Rejected { fitness } => {
                panic!("test Deep must type-check: {:?}", fitness.errors)
            }
        };
        complete_checks(analysis, SemanticContext::Isolated)
            .expect("test Deep must pass semantic checks")
    }

    fn lowered(dag: Dag) -> LoweredProgram {
        LoweredProgram {
            dag,
            rootless_defs: BTreeSet::new(),
        }
    }

    #[test]
    fn strict_successful_empty_dag_rejects_nonempty_tensor_root_names() {
        let checked = checked_compilation(
            "(def {} identity (fn {} (params {} (x {type: (t-tensor {} (d-name {} n) (t-prim {} f32))})) (var {} x)))",
        );

        let error =
            finish_isolated_lowering(checked, LoweringMode::Strict, Ok(lowered(Dag::new())))
                .expect_err("strict successful lowering must use exact root alignment");

        assert!(matches!(
            error,
            CoreLowerError::RootCount {
                context: RootCountContext::Program,
                expected: 1,
                actual: 0,
            }
        ));
    }

    #[test]
    fn successful_empty_dag_aligns_empty_tensor_root_names() {
        let checked = checked_compilation("(def {} label (lit {} \"host only\"))");

        let lowered =
            finish_isolated_lowering(checked, LoweringMode::Strict, Ok(lowered(Dag::new())))
                .expect("empty names and empty roots must align");

        assert!(lowered.dag().roots().is_empty());
        assert!(lowered.named_roots().is_empty());
    }

    #[test]
    fn selected_host_backend_accepts_a_successful_empty_dag() {
        let checked = checked_compilation(
            "(def {} identity (fn {} (params {} (x {type: (t-tensor {} (d-name {} n) (t-prim {} f32))})) (var {} x)))",
        );

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
        let checked = checked_compilation(
            "(def {} identity (fn {} (params {} (x {type: (t-tensor {} (d-name {} n) (t-prim {} f32))})) (var {} x)))",
        );
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
    fn contextual_host_backend_accepts_a_nonfatal_lower_rejection() {
        let checked = checked_compilation(
            "(def {} identity (fn {} (params {} (x {type: (t-tensor {} (d-name {} n) (t-prim {} f32))})) (var {} x)))",
        );
        let tensor_names = checked.root_metadata().tensor_names().clone();
        assert!(
            !tensor_names.is_empty(),
            "the fixture must carry a tensor root name so the arm is not vacuous"
        );
        let diagnostic = LowerDiagnostic {
            message: "host backend required".to_string(),
            span: None,
            span_id: None,
            fatal: false,
        };

        let lowered = finish_contextual_lowering(
            checked,
            &Dag::new(),
            LoweringMode::AllowHostBackend,
            Err(diagnostic),
            &tensor_names,
        )
        .expect("contextual host backend accepts a nonfatal rejection, matching isolated");

        assert!(lowered.dag().roots().is_empty());
        assert!(lowered.named_roots().is_empty());
    }

    #[test]
    fn contextual_host_only_rejects_a_nonfatal_rejection_with_tensor_names() {
        let checked = checked_compilation(
            "(def {} identity (fn {} (params {} (x {type: (t-tensor {} (d-name {} n) (t-prim {} f32))})) (var {} x)))",
        );
        let tensor_names = checked.root_metadata().tensor_names().clone();
        let diagnostic = LowerDiagnostic {
            message: "host backend required".to_string(),
            span: None,
            span_id: None,
            fatal: false,
        };

        let error = finish_contextual_lowering(
            checked,
            &Dag::new(),
            LoweringMode::AllowHostOnly,
            Err(diagnostic),
            &tensor_names,
        )
        .expect_err("host-only keeps its tensor-name guard; only host-backend is permissive");

        assert!(matches!(error, CoreLowerError::Lower(_)));
    }

    #[test]
    fn declared_root_names_drop_reported_rootless_defs() {
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

    #[test]
    fn declared_root_names_remain_when_no_defs_are_rootless() {
        let names = TensorRootNames(vec![IrName::new("ho_a"), IrName::new("sumsq")]);

        let kept = names.without(&BTreeSet::new());
        let kept: Vec<&str> = kept.iter().map(IrName::as_str).collect();
        assert_eq!(kept, ["ho_a", "sumsq"]);
    }

    #[test]
    fn isolated_named_roots_accept_exact_alignment() {
        let names = TensorRootNames(vec![IrName::new("only")]);
        let roots = NamedRoots::aligned(&names, &[NodeId(7)], RootCountContext::Program)
            .expect("isolated names and roots have equal counts");

        assert_eq!(roots.get(&IrName::new("only")), Some(&NodeId(7)));
    }

    #[test]
    fn contextual_named_roots_accept_exact_alignment() {
        let names = TensorRootNames(vec![IrName::new("new_only")]);
        let roots = NamedRoots::aligned(&names, &[NodeId(11)], RootCountContext::NewCode)
            .expect("contextual names and roots have equal counts");

        assert_eq!(roots.get(&IrName::new("new_only")), Some(&NodeId(11)));
    }

    #[test]
    fn contextual_named_roots_reject_a_count_mismatch() {
        let names = TensorRootNames(vec![IrName::new("first"), IrName::new("second")]);
        let error = NamedRoots::aligned(&names, &[NodeId(7)], RootCountContext::NewCode)
            .expect_err("contextual counts must match before map construction");

        assert!(matches!(
            error,
            CoreLowerError::RootCount {
                context: RootCountContext::NewCode,
                expected: 2,
                actual: 1,
            }
        ));
    }

    #[test]
    fn named_roots_reject_a_count_mismatch_without_a_partial_map() {
        let names = TensorRootNames(vec![IrName::new("first"), IrName::new("second")]);
        let error = NamedRoots::aligned(&names, &[NodeId(7)], RootCountContext::Program)
            .expect_err("different counts must reject before map construction");

        assert!(matches!(
            error,
            CoreLowerError::RootCount {
                context: RootCountContext::Program,
                expected: 2,
                actual: 1,
            }
        ));
    }
}
