//! Opt-in, in-memory observations of library lowering. Not a certificate.
//!
//! Existing `Dag` carriers are cloned at the actual pass boundaries, without
//! numeric conversion or a second invocation of a transformation. This module
//! intentionally defines no serialization format. See the trace contract in
//! `spec/design/chelis_hull_design_spec.md` for the remaining evidence obligations.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::dag::{Dag, NodeId, TensorType};
use crate::grad::GradResult;
use crate::lower::{LowerDiagnostic, LoweredLibrary};
use chelis_types::CheckedProgram;
use chelis_unord::UnordMap;

/// A trace-local lowering context identity, not a node or cross-run identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextKind {
    Library,
    Helper,
    Gradient,
    Vmap,
    VmapGradient,
}

#[derive(Debug, Clone)]
pub struct Context {
    pub parent: Option<ContextId>,
    pub kind: ContextKind,
}

/// An actual ordinary `grad_dag_checked` call. IDs are graph-local, except
/// `gradients` keys, which refer to `forward` and map to `backward` results.
#[derive(Debug, Clone)]
pub struct Gradient {
    pub context: ContextId,
    pub forward: Dag,
    pub output: NodeId,
    pub wrt: Vec<NodeId>,
    pub backward: Dag,
    pub backward_output: NodeId,
    /// Missing entries retain the raw AD pass's result. Source-level zero
    /// materialization happens later and is not silently attributed to this pass.
    pub gradients: BTreeMap<NodeId, NodeId>,
    /// Filled after this invocation is spliced and its result packed. `None`
    /// is an incomplete observation, never evidence of a completed application.
    pub application: Option<Application>,
}

/// The actual returned structure; leaves refer to `Application.after_packing`.
/// An empty tuple retains a discrete/unit cotangent, not a missing tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Node(NodeId),
    Tuple(Vec<Value>),
    Adt {
        ctor: String,
        field_names: Option<Vec<String>>,
        fields: Vec<Value>,
    },
}

/// The caller is the parent of the enclosing `Gradient.context`. Specialization
/// and splice maps are observations to check, not trusted semantic equalities.
#[derive(Debug, Clone)]
pub struct Application {
    pub formal_types: Vec<TensorType>,
    pub actual_types: Vec<TensorType>,
    pub specialized: Dag,
    /// Formal load names to IDs in `before_splice`, including captured bindings.
    pub arguments: BTreeMap<String, NodeId>,
    pub wrt_actuals: Vec<NodeId>,
    /// Specialized backward IDs to IDs in the caller's `after_splice`.
    pub remap: BTreeMap<NodeId, NodeId>,
    pub before_splice: Dag,
    pub after_splice: Dag,
    /// Includes shaped-zero materialization and final reuse hints.
    pub after_packing: Dag,
    pub result: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryKind {
    /// Library classification left definitions for another lowering entry.
    /// Their names remain in the returned library's `lowered_names` table.
    UnloweredDefinitions,
    UnresolvedCallableGradient,
    Vmap,
    VmapGradient,
}

/// A path outside the ordinary-grad observation, not a compiler rejection.
#[derive(Debug, Clone)]
pub struct Boundary {
    pub context: ContextId,
    pub kind: BoundaryKind,
}

/// The actual library-level normalization passes, before any backend emission.
/// These maps are observations to be checked, not trusted correspondence proofs.
#[derive(Debug, Clone)]
pub struct Normalization {
    pub before_dce: Dag,
    pub after_dce: Dag,
    pub after_copies: Dag,
    pub after_drops: Dag,
    pub dce_remap: BTreeMap<NodeId, NodeId>,
    pub copy_remap: BTreeMap<NodeId, NodeId>,
}

#[derive(Debug, Clone)]
pub struct LoweringTrace {
    /// Indexing this vector by `ContextId.0` resolves each context's parent.
    pub contexts: Vec<Context>,
    pub gradients: Vec<Gradient>,
    pub boundaries: Vec<Boundary>,
    pub normalization: Normalization,
}

/// One successful helper's owned observations. This value is retained in the
/// same private helper product as its graph and is only exposed by borrowed
/// host-plan accessors.
#[derive(Debug, Clone)]
pub struct HelperLoweringTrace {
    pub lowering: LoweringTrace,
    /// The device of every Resource handler this lowering entered, its
    /// gradient and `vmap` subcontexts included, in lowering order.
    pub requirements: Vec<String>,
    /// Every root after helper-local result packing and before normalization,
    /// in ABI order.
    pub packed_roots: Vec<NodeId>,
    /// The helper result structure whose leaves are `packed_roots` entries.
    pub packed_result: Value,
    /// The actual output of the later host dimension-rebinding boundary.
    /// `None` is explicit when this trace did not pass through that boundary
    /// (for example, a hostless named-entry plan).
    pub after_dimension_rebinding: Option<Dag>,
}

impl HelperLoweringTrace {
    pub(crate) fn record_dimension_rebinding(&mut self, dag: &Dag) {
        assert!(self.after_dimension_rebinding.is_none());
        self.after_dimension_rebinding = Some(dag.clone());
    }
}

/// Run the ordinary library lowering with explicit snapshot collection.
/// Failure returns the ordinary diagnostic, never a partial successful trace.
pub fn try_lower_program_to_library_with_trace(
    program: &CheckedProgram,
) -> Result<(LoweredLibrary, LoweringTrace), LowerDiagnostic> {
    crate::lower::try_lower_program_to_library_with_trace(program)
}

#[derive(Default)]
struct State {
    contexts: Vec<Context>,
    gradients: Vec<Gradient>,
    boundaries: Vec<Boundary>,
    normalization: Option<Normalization>,
    requirements: Vec<String>,
    helper_result: Option<(Vec<NodeId>, Value)>,
}

/// Explicitly inherited by child contexts; never global or thread-local.
#[derive(Clone)]
pub(crate) struct Collector {
    state: Rc<RefCell<State>>,
    context: ContextId,
}

pub(crate) fn ordered(remap: &UnordMap<NodeId, NodeId>) -> BTreeMap<NodeId, NodeId> {
    remap
        .to_sorted()
        .into_iter()
        .map(|(a, b)| (*a, *b))
        .collect()
}

impl Collector {
    pub(crate) fn new() -> Self {
        let state = State {
            contexts: vec![Context {
                parent: None,
                kind: ContextKind::Library,
            }],
            ..State::default()
        };
        Self {
            state: Rc::new(RefCell::new(state)),
            context: ContextId(0),
        }
    }

    pub(crate) fn new_helper() -> Self {
        let collector = Self::new();
        collector.state.borrow_mut().contexts[0].kind = ContextKind::Helper;
        collector
    }

    pub(crate) fn finish_helper(self) -> HelperLoweringTrace {
        let requirements = std::mem::take(&mut self.state.borrow_mut().requirements);
        let (packed_roots, packed_result) = self
            .state
            .borrow_mut()
            .helper_result
            .take()
            .expect("successful helper lowering records its packed result");
        HelperLoweringTrace {
            lowering: self.finish(),
            requirements,
            packed_roots,
            packed_result,
            after_dimension_rebinding: None,
        }
    }

    pub(crate) fn helper_result(&self, roots: &[NodeId], result: Value) {
        let mut state = self.state.borrow_mut();
        assert!(state.helper_result.is_none());
        state.helper_result = Some((roots.to_vec(), result));
    }

    pub(crate) fn child(&self, kind: ContextKind) -> Self {
        let mut state = self.state.borrow_mut();
        let context = ContextId(state.contexts.len());
        state.contexts.push(Context {
            parent: Some(self.context),
            kind,
        });
        Self {
            state: self.state.clone(),
            context,
        }
    }

    pub(crate) fn boundary(&self, kind: BoundaryKind) {
        self.state.borrow_mut().boundaries.push(Boundary {
            context: self.context,
            kind,
        });
    }

    pub(crate) fn gradient(
        &self,
        forward: &Dag,
        output: NodeId,
        wrt: &[NodeId],
        result: &GradResult,
    ) -> usize {
        let mut state = self.state.borrow_mut();
        let index = state.gradients.len();
        state.gradients.push(Gradient {
            context: self.context,
            forward: forward.clone(),
            output,
            wrt: wrt.to_vec(),
            backward: result.dag.clone(),
            backward_output: result.output_node,
            gradients: ordered(&result.grad_nodes),
            application: None,
        });
        index
    }

    pub(crate) fn application(&self, gradient: usize, application: Application) {
        let mut state = self.state.borrow_mut();
        let gradient = &mut state.gradients[gradient];
        assert_eq!(gradient.context, self.context);
        assert!(gradient.application.is_none());
        gradient.application = Some(application);
    }

    /// Record a Resource handler's device as this lowering enters it.
    pub(crate) fn requirement(&self, device: String) {
        self.state.borrow_mut().requirements.push(device);
    }

    pub(crate) fn normalization(
        &self,
        before_dce: &Dag,
        after_dce: &Dag,
        after_copies: Dag,
        after_drops: &Dag,
        dce_remap: &UnordMap<NodeId, NodeId>,
        copy_remap: &UnordMap<NodeId, NodeId>,
    ) {
        self.state.borrow_mut().normalization = Some(Normalization {
            before_dce: before_dce.clone(),
            after_dce: after_dce.clone(),
            after_copies,
            after_drops: after_drops.clone(),
            dce_remap: ordered(dce_remap),
            copy_remap: ordered(copy_remap),
        });
    }

    pub(crate) fn finish(self) -> LoweringTrace {
        let mut state = self.state.borrow_mut();
        LoweringTrace {
            contexts: std::mem::take(&mut state.contexts),
            gradients: std::mem::take(&mut state.gradients),
            boundaries: std::mem::take(&mut state.boundaries),
            normalization: state
                .normalization
                .take()
                .expect("successful library lowering records normalization"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{DimInfo, RiscOp, TensorType};
    use chelis_types::types::Prim;

    #[test]
    fn retained_helper_trace_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HelperLoweringTrace>();
    }

    // This checks the collector's metadata fidelity, not host-entry coverage.
    // Source-driven library-path tests live in tests/lowering_trace.rs.
    #[test]
    fn snapshots_retain_shape_dependencies_and_spans_without_aliasing() {
        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load {
                name: "shape_source".into(),
            },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("n".into(), None)],
                precision: Prim::F32,
            },
            Some("input-span".into()),
        );
        let output = dag.add_node(
            RiscOp::synth_const(Prim::F32, -0.0),
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::F32,
            },
            Some("output-span".into()),
        );
        dag.add_shape_dep(output, input);
        dag.node_mut(output).unwrap().merged_spans = vec!["merged-span".into()];
        dag.add_root(output);
        let result = crate::grad::grad_dag_checked(&dag, output, &[input]).unwrap();
        let collector = Collector::new();
        collector.gradient(&dag, output, &[input], &result);
        let captured = &collector.state.borrow().gradients[0];
        let before = bincode::serialize(&dag).unwrap();
        assert_eq!(bincode::serialize(&captured.forward).unwrap(), before);
        assert_eq!(
            bincode::serialize(&captured.backward).unwrap(),
            bincode::serialize(&result.dag).unwrap()
        );
        assert_eq!(
            captured.forward.get(output).unwrap().shape_deps,
            vec![input]
        );
        // A missing metadata edge or a later mutation cannot rewrite history.
        dag.node_mut(output).unwrap().shape_deps.clear();
        dag.node_mut(output).unwrap().span_id = None;
        assert_ne!(bincode::serialize(&dag).unwrap(), before);
        assert_eq!(bincode::serialize(&captured.forward).unwrap(), before);
    }
}
