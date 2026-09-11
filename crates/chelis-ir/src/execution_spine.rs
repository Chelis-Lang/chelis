//! Source events are recorded before graph rewrites, independently of the
//! executable schedule. A value-free handler still has two source events.

use std::collections::{BTreeMap, BTreeSet};

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

// These established public carriers remain random-only and `Copy`. The full
// sidecar is allocated only if a Resource requirement is actually recorded.
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

/// Full-source identities are trace-local and deliberately distinct from the
/// established random-only [`OccurrenceId`] namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct FullOccurrenceId(pub(crate) usize);

#[derive(Debug, Clone, PartialEq, Eq)]
enum FullSourceKind {
    Legacy(Occurrence),
    Requirement(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FullOccurrence {
    id: FullOccurrenceId,
    kind: FullSourceKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FullStep {
    Node(NodeId),
    Control {
        occurrence: FullOccurrenceId,
        legacy_occurrence: OccurrenceId,
        control: Control,
    },
    Requirement {
        occurrence: FullOccurrenceId,
    },
}

#[derive(Debug, Clone)]
struct FullSpine {
    steps: Vec<FullStep>,
    source: Vec<FullOccurrence>,
    next_occurrence: usize,
}

impl FullSpine {
    fn from_legacy(steps: &[Step], source: &[Occurrence]) -> Self {
        let ids = source
            .iter()
            .enumerate()
            .map(|(id, event)| (event.id, FullOccurrenceId(id)))
            .collect::<BTreeMap<_, _>>();
        Self {
            steps: steps
                .iter()
                .map(|step| match *step {
                    Step::Node(node) => FullStep::Node(node),
                    Step::Control {
                        occurrence,
                        control,
                    } => FullStep::Control {
                        occurrence: ids[&occurrence],
                        legacy_occurrence: occurrence,
                        control,
                    },
                })
                .collect(),
            source: source
                .iter()
                .enumerate()
                .map(|(id, event)| FullOccurrence {
                    id: FullOccurrenceId(id),
                    kind: FullSourceKind::Legacy(*event),
                })
                .collect(),
            next_occurrence: source.len(),
        }
    }

    fn occurrence(&mut self, kind: FullSourceKind) -> FullOccurrenceId {
        let id = FullOccurrenceId(self.next_occurrence);
        self.next_occurrence += 1;
        self.source.push(FullOccurrence { id, kind });
        id
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Spine {
    pub(crate) steps: Vec<Step>,
    // This census is created at lowering, not reconstructed from `steps` or
    // the final graph. Transformations may map identities, not invent events.
    pub(crate) source: Vec<Occurrence>,
    next_occurrence: usize,
    recorded_nodes: usize,
    // Boxed so ordinary Random-only plans retain their prior vector storage
    // and do not pay for a duplicate census/schedule.
    full: Option<Box<FullSpine>>,
    #[cfg(test)]
    legacy_projection_rebuilds: usize,
}

impl Spine {
    pub(crate) fn occurrence_count(&self) -> usize {
        self.next_occurrence
    }

    pub(crate) fn full_occurrence_count(&self) -> usize {
        self.full
            .as_ref()
            .map_or(self.next_occurrence, |full| full.next_occurrence)
    }

    fn ensure_full(&mut self) -> &mut FullSpine {
        self.full
            .get_or_insert_with(|| Box::new(FullSpine::from_legacy(&self.steps, &self.source)))
    }

    fn refresh_legacy(&mut self) {
        let Some(full) = &self.full else {
            return;
        };
        #[cfg(test)]
        {
            self.legacy_projection_rebuilds += 1;
        }
        self.steps = full
            .steps
            .iter()
            .filter_map(|step| match step {
                FullStep::Node(node) => Some(Step::Node(*node)),
                FullStep::Control {
                    legacy_occurrence,
                    control,
                    ..
                } => Some(Step::Control {
                    occurrence: *legacy_occurrence,
                    control: *control,
                }),
                FullStep::Requirement { .. } => None,
            })
            .collect();
        self.source = full
            .source
            .iter()
            .filter_map(|event| match event.kind {
                FullSourceKind::Legacy(event) => Some(event),
                FullSourceKind::Requirement(_) => None,
            })
            .collect();
    }

    fn validate_legacy_projection(&self) -> Result<(), String> {
        let Some(_) = self.full else {
            return Ok(());
        };
        let mut projected = self.clone();
        projected.refresh_legacy();
        if projected.steps != self.steps || projected.source != self.source {
            return Err("legacy execution view diverges from the full source spine".into());
        }
        Ok(())
    }

    fn occurrence(&mut self, kind: SourceKind) -> (OccurrenceId, Option<FullOccurrenceId>) {
        let id = OccurrenceId(self.next_occurrence);
        self.next_occurrence += 1;
        let event = Occurrence { id, kind };
        let full = self
            .full
            .as_mut()
            .map(|full| full.occurrence(FullSourceKind::Legacy(event)));
        self.source.push(event);
        (id, full)
    }

    /// Flush the actual append-only lowering segment before a source cut.
    /// Loads are supplied by the enclosing region and inserted on completion;
    /// unused formal loads may legitimately disappear during AD pruning.
    pub(crate) fn record_nodes(&mut self, dag: &Dag) {
        assert!(self.recorded_nodes <= dag.len());
        for node in &dag.nodes()[self.recorded_nodes..] {
            if !matches!(node.op, RiscOp::Load { .. }) {
                self.steps.push(Step::Node(node.id));
                if let Some(full) = &mut self.full {
                    full.steps.push(FullStep::Node(node.id));
                }
            }
        }
        self.recorded_nodes = dag.len();
    }

    pub(crate) fn forward(&mut self, node: NodeId, draw: DrawId, scope: ScopeId) {
        self.occurrence(SourceKind::Forward { node, draw, scope });
    }

    pub(crate) fn control(&mut self, dag: &Dag, control: Control) {
        self.record_nodes(dag);
        let (legacy_occurrence, full_occurrence) = self.occurrence(SourceKind::Control(control));
        self.steps.push(Step::Control {
            occurrence: legacy_occurrence,
            control,
        });
        if let (Some(full), Some(occurrence)) = (&mut self.full, full_occurrence) {
            full.steps.push(FullStep::Control {
                occurrence,
                legacy_occurrence,
                control,
            });
        }
    }

    pub(crate) fn require(&mut self, dag: &Dag, device: String) {
        self.record_nodes(dag);
        let full = self.ensure_full();
        let occurrence = full.occurrence(FullSourceKind::Requirement(device));
        full.steps.push(FullStep::Requirement { occurrence });
    }

    pub(crate) fn nodes(&self) -> Vec<NodeId> {
        if let Some(full) = &self.full {
            full.steps
                .iter()
                .filter_map(|step| match step {
                    FullStep::Node(node) => Some(*node),
                    FullStep::Control { .. } | FullStep::Requirement { .. } => None,
                })
                .collect()
        } else {
            self.steps
                .iter()
                .filter_map(|step| match step {
                    Step::Node(node) => Some(*node),
                    Step::Control { .. } => None,
                })
                .collect()
        }
    }

    /// Only an owning pass may remove nodes whose removal it authorizes (for
    /// example standalone-library terminal Drops). Source events are retained.
    pub(crate) fn retain_nodes(&mut self, mut keep: impl FnMut(NodeId) -> bool) {
        if let Some(full) = &mut self.full {
            full.steps.retain(|step| match step {
                FullStep::Node(node) => keep(*node),
                FullStep::Control { .. } | FullStep::Requirement { .. } => true,
            });
            self.refresh_legacy();
        } else {
            self.steps.retain(|step| match step {
                Step::Node(node) => keep(*node),
                Step::Control { .. } => true,
            });
        }
    }

    pub(crate) fn remap(&mut self, remap: &UnordMap<NodeId, NodeId>) -> Result<(), String> {
        let mapped = |node: NodeId| {
            remap
                .get(&node)
                .copied()
                .ok_or_else(|| format!("execution spine lost source node {}", node.0))
        };
        if let Some(full) = &mut self.full {
            for step in &mut full.steps {
                if let FullStep::Node(node) = step {
                    *node = mapped(*node)?;
                }
            }
            for occurrence in &mut full.source {
                if let FullSourceKind::Legacy(Occurrence {
                    kind: SourceKind::Forward { node, .. },
                    ..
                }) = &mut occurrence.kind
                {
                    *node = mapped(*node)?;
                }
            }
            self.refresh_legacy();
        } else {
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
        }
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
        if let Some(full) = &mut self.full {
            let mut next = 0;
            let mut steps = Vec::new();
            for step in &full.steps {
                match *step {
                    FullStep::Node(node) => {
                        if node.0 < next || node.0 >= dag.len() {
                            return Err(
                                "execution rewrite reordered an existing source node".into()
                            );
                        }
                        steps.extend((next..=node.0).map(|index| FullStep::Node(NodeId(index))));
                        next = node.0 + 1;
                    }
                    event => steps.push(event),
                }
            }
            steps.extend((next..dag.len()).map(|index| FullStep::Node(NodeId(index))));
            full.steps = steps;
            self.refresh_legacy();
        } else {
            let mut next = 0;
            let mut steps = Vec::new();
            for step in &self.steps {
                match *step {
                    Step::Node(node) => {
                        if node.0 < next || node.0 >= dag.len() {
                            return Err(
                                "execution rewrite reordered an existing source node".into()
                            );
                        }
                        steps.extend((next..=node.0).map(|index| Step::Node(NodeId(index))));
                        next = node.0 + 1;
                    }
                    control => steps.push(control),
                }
            }
            steps.extend((next..dag.len()).map(|index| Step::Node(NodeId(index))));
            self.steps = steps;
        }
        self.recorded_nodes = dag.len();
        Ok(())
    }

    pub(crate) fn selected(
        &self,
        remap: &UnordMap<NodeId, NodeId>,
        occurrences: &BTreeSet<FullOccurrenceId>,
    ) -> Result<Self, String> {
        let Some(_) = self.full else {
            let legacy = occurrences.iter().map(|id| OccurrenceId(id.0)).collect();
            return self.selected_legacy(remap, &legacy);
        };
        let mut selected = self.clone();
        let full = selected.full.as_mut().expect("Resource spine sidecar");
        full.source.retain(|event| occurrences.contains(&event.id));
        if full.source.len() != occurrences.len() {
            return Err("selected declaration lost a full source occurrence".into());
        }
        full.steps.retain(|step| match step {
            FullStep::Node(node) => remap.contains_key(node),
            FullStep::Control { occurrence, .. } | FullStep::Requirement { occurrence } => {
                occurrences.contains(occurrence)
            }
        });
        selected.remap(remap)?;
        Ok(selected)
    }

    pub(crate) fn selected_legacy(
        &self,
        remap: &UnordMap<NodeId, NodeId>,
        occurrences: &BTreeSet<OccurrenceId>,
    ) -> Result<Self, String> {
        if self.full.is_some() {
            let full = self
                .full
                .as_ref()
                .expect("Resource spine sidecar")
                .source
                .iter()
                .filter_map(|event| match event.kind {
                    FullSourceKind::Legacy(legacy) if occurrences.contains(&legacy.id) => {
                        Some(event.id)
                    }
                    _ => None,
                })
                .collect();
            return self.selected(remap, &full);
        }
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
        if let Some(full) = &mut self.full {
            for step in &mut full.steps {
                if let FullStep::Control {
                    control: command, ..
                } = step
                {
                    control(command);
                }
            }
            for event in &mut full.source {
                match &mut event.kind {
                    FullSourceKind::Legacy(Occurrence {
                        kind:
                            SourceKind::Forward {
                                draw: id,
                                scope: scope_id,
                                ..
                            },
                        ..
                    }) => {
                        *id = draw(*id)?;
                        *scope_id = scope(*scope_id);
                    }
                    FullSourceKind::Legacy(Occurrence {
                        kind: SourceKind::Control(command),
                        ..
                    }) => control(command),
                    FullSourceKind::Requirement(_) => {}
                }
            }
            self.refresh_legacy();
        } else {
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
        }
        Ok(())
    }

    /// Splicing an AD child maps its formal loads to existing caller values.
    /// Those loads are not re-executed; all child source events occur once.
    pub(crate) fn append_child(&mut self, mut child: Self, dag: &Dag) -> Result<(), String> {
        if self.full.is_none() && child.full.is_none() {
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
            return self.complete(dag);
        }

        self.ensure_full();
        child.ensure_full();
        let occurrence_offset = self.next_occurrence;
        let full_offset = self.full_occurrence_count();
        let child_full = child.full.take().expect("promoted child Resource spine");
        let mut child_full = *child_full;
        child_full.steps.retain(|step| match step {
            FullStep::Node(node) => node.0 >= self.recorded_nodes,
            FullStep::Control { .. } | FullStep::Requirement { .. } => true,
        });
        for step in &mut child_full.steps {
            match step {
                FullStep::Control {
                    occurrence,
                    legacy_occurrence,
                    ..
                } => {
                    occurrence.0 += full_offset;
                    legacy_occurrence.0 += occurrence_offset;
                }
                FullStep::Requirement { occurrence } => occurrence.0 += full_offset,
                FullStep::Node(_) => {}
            }
        }
        for event in &mut child_full.source {
            event.id.0 += full_offset;
            if let FullSourceKind::Legacy(legacy) = &mut event.kind {
                legacy.id.0 += occurrence_offset;
            }
        }
        self.next_occurrence += child.next_occurrence;
        let full = self.full.as_mut().expect("promoted parent Resource spine");
        full.next_occurrence += child_full.next_occurrence;
        full.steps.extend(child_full.steps);
        full.source.extend(child_full.source);
        self.complete(dag)
    }

    pub(crate) fn validate_full(
        &self,
        mut is_forward: impl FnMut(NodeId) -> bool,
    ) -> Result<(), String> {
        let Some(full) = &self.full else {
            return Ok(());
        };
        self.validate_legacy_projection()?;
        let mut source = full.source.iter();
        let mut seen_occurrences = BTreeSet::new();
        for step in &full.steps {
            let expected_source = match step {
                FullStep::Requirement { occurrence } => Some((*occurrence, true)),
                FullStep::Control { occurrence, .. } => Some((*occurrence, false)),
                FullStep::Node(_) => None,
            };
            if let Some((id, requirement)) = expected_source {
                let event = source
                    .next()
                    .ok_or("execution has an extra full source cut")?;
                let kind_matches = matches!(
                    (&event.kind, requirement),
                    (FullSourceKind::Requirement(_), true)
                        | (
                            FullSourceKind::Legacy(Occurrence {
                                kind: SourceKind::Control(_),
                                ..
                            }),
                            false
                        )
                );
                if event.id != id || !kind_matches || !seen_occurrences.insert(id) {
                    return Err("execution cut disagrees with the full source census".into());
                }
                continue;
            }
            let FullStep::Node(node) = step else {
                unreachable!()
            };
            if is_forward(*node) {
                let event = source
                    .next()
                    .ok_or("execution has an extra full source draw")?;
                if !matches!(event.kind, FullSourceKind::Legacy(Occurrence { kind: SourceKind::Forward { node: source_node, .. }, .. }) if source_node == *node)
                    || !seen_occurrences.insert(event.id)
                {
                    return Err("execution draw disagrees with the full source census".into());
                }
            }
        }
        if source.next().is_some() {
            return Err("execution omits a full source occurrence".into());
        }
        Ok(())
    }

    #[cfg(feature = "lowering-trace")]
    pub(crate) fn full_observation(&self) -> crate::lowering_trace::FullSpineObservation {
        use crate::lowering_trace::{
            FullSourceKind as ObservedKind, FullSourceOccurrence as ObservedOccurrence,
            FullSpineObservation, FullStep as ObservedStep, SourceEventId,
        };
        let promoted;
        let full = if let Some(full) = &self.full {
            full.as_ref()
        } else {
            promoted = FullSpine::from_legacy(&self.steps, &self.source);
            &promoted
        };
        FullSpineObservation {
            steps: full
                .steps
                .iter()
                .map(|step| match step {
                    FullStep::Node(node) => ObservedStep::Node(*node),
                    FullStep::Control {
                        occurrence,
                        control,
                        ..
                    } => ObservedStep::Control {
                        occurrence: SourceEventId(occurrence.0),
                        control: *control,
                    },
                    FullStep::Requirement { occurrence } => ObservedStep::Requirement {
                        occurrence: SourceEventId(occurrence.0),
                    },
                })
                .collect(),
            source: full
                .source
                .iter()
                .map(|event| ObservedOccurrence {
                    id: SourceEventId(event.id.0),
                    kind: match &event.kind {
                        FullSourceKind::Legacy(Occurrence { kind, .. }) => match *kind {
                            SourceKind::Forward { node, draw, scope } => {
                                ObservedKind::Forward { node, draw, scope }
                            }
                            SourceKind::Control(control) => ObservedKind::Control(control),
                        },
                        FullSourceKind::Requirement(device) => {
                            ObservedKind::Requirement(device.clone())
                        }
                    },
                })
                .collect(),
        }
    }

    #[cfg(test)]
    pub(crate) fn corrupt_requirement_for_test(&mut self, corruption: usize) {
        let full = self.full.as_mut().expect("test Resource spine sidecar");
        let position = full
            .steps
            .iter()
            .position(|step| matches!(step, FullStep::Requirement { .. }))
            .expect("test spine has a Resource requirement");
        match corruption {
            0 => {
                full.steps.remove(position);
            }
            1 => {
                let event = full.steps[position];
                full.steps.insert(position, event);
            }
            2 => {
                full.steps[position] = FullStep::Requirement {
                    occurrence: FullOccurrenceId(full.next_occurrence),
                };
            }
            _ => {
                let last = full.steps.len() - 1;
                full.steps.swap(position, last);
            }
        }
        self.refresh_legacy();
    }

    #[cfg(test)]
    pub(crate) fn full_source_ids_for_test(&self) -> impl Iterator<Item = FullOccurrenceId> + '_ {
        let legacy = self
            .source
            .iter()
            .enumerate()
            .map(|(id, _)| FullOccurrenceId(id));
        self.full
            .as_ref()
            .map(|full| full.source.iter().map(|event| event.id).collect::<Vec<_>>())
            .unwrap_or_else(|| legacy.collect())
            .into_iter()
    }

    #[cfg(test)]
    pub(crate) fn legacy_projection_rebuilds_for_test(&self) -> usize {
        self.legacy_projection_rebuilds
    }

    #[cfg(test)]
    pub(crate) fn has_full_sidecar_for_test(&self) -> bool {
        self.full.is_some()
    }
}
