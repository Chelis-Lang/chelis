//! Opt-in, in-memory observations of library lowering. Not a certificate.
//!
//! Existing `Dag` carriers are cloned at the actual pass boundaries, without
//! numeric conversion or a second invocation of a transformation. This module
//! intentionally defines no serialization format. See the trace contract in
//! `spec/design/chelis_hull_design_spec.md` for the remaining evidence obligations.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::dag::{Dag, NodeId};
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
}

/// Explicitly inherited by child contexts; never global or thread-local.
#[derive(Clone)]
pub(crate) struct Collector {
    state: Rc<RefCell<State>>,
    context: ContextId,
}

fn ordered(remap: &UnordMap<NodeId, NodeId>) -> BTreeMap<NodeId, NodeId> {
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
    ) {
        self.state.borrow_mut().gradients.push(Gradient {
            context: self.context,
            forward: forward.clone(),
            output,
            wrt: wrt.to_vec(),
            backward: result.dag.clone(),
            backward_output: result.output_node,
            gradients: ordered(&result.grad_nodes),
        });
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
