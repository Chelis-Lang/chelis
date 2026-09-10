//! Source events are recorded before graph rewrites, independently of the
//! executable schedule. A value-free handler still has two source events.

use std::collections::BTreeSet;

use chelis_unord::UnordMap;

use crate::dag::{Dag, NodeId, RiscOp};
use crate::evaluation::{DrawId, ScopeId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct OccurrenceId(pub(crate) usize);

impl OccurrenceId {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    Enter { scope: ScopeId, seed: u64 },
    Leave { scope: ScopeId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Node(NodeId),
    Control {
        occurrence: OccurrenceId,
        control: Control,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Forward {
        node: NodeId,
        draw: DrawId,
        scope: ScopeId,
    },
    Control(Control),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Occurrence {
    pub id: OccurrenceId,
    pub kind: SourceKind,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Spine {
    pub(crate) steps: Vec<Step>,
    // This census is created at lowering, not reconstructed from `steps` or
    // the final graph. Transformations may map identities, not invent events.
    pub(crate) source: Vec<Occurrence>,
    next_occurrence: usize,
    recorded_nodes: usize,
}

impl Spine {
    pub(crate) fn occurrence_count(&self) -> usize {
        self.next_occurrence
    }

    fn occurrence(&mut self, kind: SourceKind) -> OccurrenceId {
        let id = OccurrenceId(self.next_occurrence);
        self.next_occurrence += 1;
        self.source.push(Occurrence { id, kind });
        id
    }

    /// Flush the actual append-only lowering segment before a source cut.
    /// Loads are supplied by the enclosing region and inserted on completion;
    /// unused formal loads may legitimately disappear during AD pruning.
    pub(crate) fn record_nodes(&mut self, dag: &Dag) {
        assert!(self.recorded_nodes <= dag.len());
        self.steps.extend(
            dag.nodes()[self.recorded_nodes..]
                .iter()
                .filter(|node| !matches!(node.op, RiscOp::Load { .. }))
                .map(|node| Step::Node(node.id)),
        );
        self.recorded_nodes = dag.len();
    }

    pub(crate) fn forward(&mut self, node: NodeId, draw: DrawId, scope: ScopeId) {
        self.occurrence(SourceKind::Forward { node, draw, scope });
    }

    pub(crate) fn control(&mut self, dag: &Dag, control: Control) {
        self.record_nodes(dag);
        let occurrence = self.occurrence(SourceKind::Control(control));
        self.steps.push(Step::Control {
            occurrence,
            control,
        });
    }

    pub(crate) fn nodes(&self) -> Vec<NodeId> {
        self.steps
            .iter()
            .filter_map(|step| match step {
                Step::Node(node) => Some(*node),
                Step::Control { .. } => None,
            })
            .collect()
    }

    /// Only an owning pass may remove nodes whose removal it authorizes (for
    /// example standalone-library terminal Drops). Source events are retained.
    pub(crate) fn retain_nodes(&mut self, mut keep: impl FnMut(NodeId) -> bool) {
        self.steps.retain(|step| match step {
            Step::Node(node) => keep(*node),
            Step::Control { .. } => true,
        });
    }

    pub(crate) fn remap(&mut self, remap: &UnordMap<NodeId, NodeId>) -> Result<(), String> {
        let mapped = |node: NodeId| {
            remap
                .get(&node)
                .copied()
                .ok_or_else(|| format!("execution spine lost source node {}", node.0))
        };
        for step in &mut self.steps {
            if let Step::Node(node) = step {
                *node = mapped(*node)?;
            }
        }
        for occurrence in &mut self.source {
            if let SourceKind::Forward { node, .. } = &mut occurrence.kind {
                *node = mapped(*node)?;
            }
        }
        // The image's exclusive upper bound is zero for an empty graph.
        self.recorded_nodes = remap
            .to_sorted()
            .iter()
            .fold(0, |bound, (_, node)| bound.max(node.0 + 1));
        Ok(())
    }

    /// Copy insertion preserves existing node order. Weave new nodes before
    /// their next retained node, without sorting or reconstructing controls.
    /// A source cut stays before the copies introduced for its next consumer.
    /// Terminal Drops remain after the final source cut.
    pub(crate) fn complete(&mut self, dag: &Dag) -> Result<(), String> {
        let mut next = 0;
        let mut steps = Vec::new();
        for step in &self.steps {
            match *step {
                Step::Node(node) => {
                    if node.0 < next || node.0 >= dag.len() {
                        return Err("execution rewrite reordered an existing source node".into());
                    }
                    steps.extend((next..=node.0).map(|index| Step::Node(NodeId(index))));
                    next = node.0 + 1;
                }
                control => steps.push(control),
            }
        }
        steps.extend((next..dag.len()).map(|index| Step::Node(NodeId(index))));
        self.steps = steps;
        self.recorded_nodes = dag.len();
        Ok(())
    }

    pub(crate) fn selected(
        &self,
        remap: &UnordMap<NodeId, NodeId>,
        occurrences: &BTreeSet<OccurrenceId>,
    ) -> Result<Self, String> {
        let mut selected = self.clone();
        selected
            .source
            .retain(|event| occurrences.contains(&event.id));
        if selected.source.len() != occurrences.len() {
            return Err("selected declaration lost a source occurrence".into());
        }
        selected.steps.retain(|step| match step {
            Step::Node(node) => remap.contains_key(node),
            Step::Control { occurrence, .. } => occurrences.contains(occurrence),
        });
        selected.remap(remap)?;
        Ok(selected)
    }

    pub(crate) fn rebase(
        &mut self,
        mut draw: impl FnMut(DrawId) -> Result<DrawId, String>,
        scope: impl Fn(ScopeId) -> ScopeId,
    ) -> Result<(), String> {
        let control = |control: &mut Control| match control {
            Control::Enter { scope: id, .. } | Control::Leave { scope: id } => *id = scope(*id),
        };
        for step in &mut self.steps {
            if let Step::Control {
                control: command, ..
            } = step
            {
                control(command);
            }
        }
        for event in &mut self.source {
            match &mut event.kind {
                SourceKind::Forward {
                    draw: id,
                    scope: scope_id,
                    ..
                } => {
                    *id = draw(*id)?;
                    *scope_id = scope(*scope_id);
                }
                SourceKind::Control(command) => control(command),
            }
        }
        Ok(())
    }

    /// Splicing an AD child maps its formal loads to existing caller values.
    /// Those loads are not re-executed; all child source controls occur once.
    pub(crate) fn append_child(&mut self, mut child: Self, dag: &Dag) -> Result<(), String> {
        child.steps.retain(|step| match step {
            Step::Node(node) => node.0 >= self.recorded_nodes,
            Step::Control { .. } => true,
        });
        let offset = self.next_occurrence;
        for step in &mut child.steps {
            if let Step::Control { occurrence, .. } = step {
                occurrence.0 += offset;
            }
        }
        for event in &mut child.source {
            event.id.0 += offset;
        }
        self.next_occurrence += child.next_occurrence;
        self.steps.extend(child.steps);
        self.source.extend(child.source);
        self.complete(dag)
    }
}
