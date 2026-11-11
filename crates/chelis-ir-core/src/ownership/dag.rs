use super::OwnershipError;
use crate::prelude::*;
use alloc::collections::BTreeMap;
use chelis_dag_core::dag::{Dag, DagNode, NodeId, RiscOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DagOwnerOrigin {
    BorrowedLoad,
    OwnedProducer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DagOwnerState {
    BorrowedLive,
    OwnedLive,
    OwnedTerminal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DagDirective {
    BorrowLoad { node: NodeId },
    Produce { node: NodeId, borrows: Vec<NodeId> },
    Clone { node: NodeId, source: NodeId },
    CloneProduce { node: NodeId, source: NodeId },
    MoveProduce { node: NodeId, source: NodeId },
    CloneStore { node: NodeId, source: NodeId },
    MoveStore { node: NodeId, source: NodeId },
    StoreRoot { node: NodeId },
    BorrowedDrop { node: NodeId, source: NodeId },
    OwnedDrop { node: NodeId, source: NodeId },
    Root { source: NodeId },
    RootClone { source: NodeId },
    ScopeDrop { source: NodeId },
}

impl DagDirective {
    pub fn node(&self) -> Option<NodeId> {
        match *self {
            Self::BorrowLoad { node }
            | Self::Produce { node, .. }
            | Self::Clone { node, .. }
            | Self::CloneProduce { node, .. }
            | Self::MoveProduce { node, .. }
            | Self::CloneStore { node, .. }
            | Self::MoveStore { node, .. }
            | Self::StoreRoot { node }
            | Self::BorrowedDrop { node, .. }
            | Self::OwnedDrop { node, .. } => Some(node),
            Self::Root { .. } | Self::RootClone { .. } | Self::ScopeDrop { .. } => None,
        }
    }
}

#[derive(Debug)]
pub struct DagOwnershipPlan {
    pub owners: BTreeMap<NodeId, DagOwnerOrigin>,
    pub directives: Vec<DagDirective>,
}

impl DagOwnershipPlan {
    pub fn lower(dag: &Dag) -> Result<Self, OwnershipError> {
        let structural_errors = chelis_dag_core::verify::verify_ownership_input(dag);
        if !structural_errors.is_empty() {
            return Err(OwnershipError::LoweringInvariant {
                unit: "dag".to_string(),
                detail: structural_errors.join("; "),
            });
        }
        let mut owners = BTreeMap::new();
        let mut states = BTreeMap::new();
        let mut directives = Vec::new();
        for node in dag.nodes() {
            validate_dag_dependencies(dag, node)?;
            match &node.op {
                RiscOp::Load { .. } => {
                    require_dag_arity(node.id, "load", 0, node.inputs.len())?;
                    owners.insert(node.id, DagOwnerOrigin::BorrowedLoad);
                    states.insert(node.id, DagOwnerState::BorrowedLive);
                    directives.push(DagDirective::BorrowLoad { node: node.id });
                }
                RiscOp::Copy => {
                    require_dag_arity(node.id, "copy", 1, node.inputs.len())?;
                    require_live_dag_owner(&states, node.inputs[0], node.id)?;
                    owners.insert(node.id, DagOwnerOrigin::OwnedProducer);
                    states.insert(node.id, DagOwnerState::OwnedLive);
                    directives.push(DagDirective::Clone {
                        node: node.id,
                        source: node.inputs[0],
                    });
                }
                RiscOp::Drop => {
                    require_dag_arity(node.id, "drop", 1, node.inputs.len())?;
                    match states.get(&node.inputs[0]).copied() {
                        Some(DagOwnerState::BorrowedLive) => {
                            directives.push(DagDirective::BorrowedDrop {
                                node: node.id,
                                source: node.inputs[0],
                            });
                        }
                        Some(DagOwnerState::OwnedLive) => {
                            consume_dag_owner(&mut states, node.inputs[0], node.id)?;
                            directives.push(DagDirective::OwnedDrop {
                                node: node.id,
                                source: node.inputs[0],
                            });
                        }
                        Some(DagOwnerState::OwnedTerminal) => {
                            return Err(OwnershipError::DagDuplicateTerminal {
                                owner: node.inputs[0].0,
                            });
                        }
                        None => {
                            return Err(OwnershipError::DagInput {
                                node: node.id.0,
                                input: node.inputs[0].0,
                            });
                        }
                    }
                }
                RiscOp::Realize => {
                    require_dag_arity(node.id, "realize", 1, node.inputs.len())?;
                    let source_state = require_live_dag_owner(&states, node.inputs[0], node.id)?;
                    owners.insert(node.id, DagOwnerOrigin::OwnedProducer);
                    states.insert(node.id, DagOwnerState::OwnedLive);
                    match source_state {
                        DagOwnerState::BorrowedLive => {
                            directives.push(DagDirective::CloneProduce {
                                node: node.id,
                                source: node.inputs[0],
                            });
                        }
                        DagOwnerState::OwnedLive
                            if dag_owner_used_after(dag, node.inputs[0], node.id) =>
                        {
                            directives.push(DagDirective::CloneProduce {
                                node: node.id,
                                source: node.inputs[0],
                            });
                        }
                        DagOwnerState::OwnedLive => {
                            consume_dag_owner(&mut states, node.inputs[0], node.id)?;
                            directives.push(DagDirective::MoveProduce {
                                node: node.id,
                                source: node.inputs[0],
                            });
                        }
                        DagOwnerState::OwnedTerminal => {
                            unreachable!("require_live_dag_owner rejects terminal owners")
                        }
                    }
                }
                RiscOp::Store { .. } => {
                    require_dag_arity(node.id, "store", 1, node.inputs.len())?;
                    let source_state = require_live_dag_owner(&states, node.inputs[0], node.id)?;
                    if source_state == DagOwnerState::BorrowedLive {
                        directives.push(DagDirective::CloneStore {
                            node: node.id,
                            source: node.inputs[0],
                        });
                    } else {
                        consume_dag_owner(&mut states, node.inputs[0], node.id)?;
                        directives.push(DagDirective::MoveStore {
                            node: node.id,
                            source: node.inputs[0],
                        });
                    }
                }
                _ => {
                    let mut borrows = node.inputs.clone();
                    // A node reads its activation to decide whether it checks.
                    for dependency in node.shape_deps.iter().chain(&node.owner.activation) {
                        if !borrows.contains(dependency) {
                            borrows.push(*dependency);
                        }
                    }
                    for source in &borrows {
                        require_live_dag_owner(&states, *source, node.id)?;
                    }
                    owners.insert(node.id, DagOwnerOrigin::OwnedProducer);
                    states.insert(node.id, DagOwnerState::OwnedLive);
                    directives.push(DagDirective::Produce {
                        node: node.id,
                        borrows,
                    });
                }
            }
        }
        for (root_index, root) in dag.roots().iter().enumerate() {
            if matches!(
                dag.get(*root).map(|node| &node.op),
                Some(RiscOp::Store { .. })
            ) {
                directives.push(DagDirective::StoreRoot { node: *root });
                continue;
            }
            let consumer = NodeId(dag.nodes().len() + root_index);
            match states.get(root).copied() {
                Some(DagOwnerState::OwnedLive) => {
                    consume_dag_owner(&mut states, *root, consumer)?;
                    directives.push(DagDirective::Root { source: *root });
                }
                Some(DagOwnerState::BorrowedLive) => {
                    directives.push(DagDirective::RootClone { source: *root });
                }
                Some(DagOwnerState::OwnedTerminal) => {
                    return Err(OwnershipError::DagDuplicateTerminal { owner: root.0 });
                }
                None => return Err(OwnershipError::DagInvalidRoot { root: root.0 }),
            }
        }
        let live_owned = states
            .iter()
            .filter_map(|(owner, state)| (*state == DagOwnerState::OwnedLive).then_some(*owner))
            .collect::<Vec<_>>();
        for owner in live_owned {
            consume_dag_owner(
                &mut states,
                owner,
                NodeId(dag.nodes().len() + dag.roots().len()),
            )?;
            directives.push(DagDirective::ScopeDrop { source: owner });
        }
        for (&owner, origin) in &owners {
            if *origin == DagOwnerOrigin::OwnedProducer
                && states.get(&owner) != Some(&DagOwnerState::OwnedTerminal)
            {
                return Err(OwnershipError::DagMissingTerminal { owner: owner.0 });
            }
        }
        Ok(Self { owners, directives })
    }

    pub fn verify(&self, dag: &Dag) -> Result<(), OwnershipError> {
        let structural_errors = chelis_dag_core::verify::verify_ownership_input(dag);
        if !structural_errors.is_empty() {
            return Err(OwnershipError::LoweringInvariant {
                unit: "dag".to_string(),
                detail: structural_errors.join("; "),
            });
        }

        let mut owners = BTreeMap::new();
        let mut states = BTreeMap::new();
        let mut cursor = 0;
        for node in dag.nodes() {
            validate_dag_dependencies(dag, node)?;
            let directive = self.directives.get(cursor).ok_or_else(|| {
                dag_directive_error(format!("node n{} has no directive", node.id.0))
            })?;
            cursor += 1;
            match &node.op {
                RiscOp::Load { .. } => {
                    require_dag_arity(node.id, "load", 0, node.inputs.len())?;
                    require_exact_dag_directive(
                        directive,
                        &DagDirective::BorrowLoad { node: node.id },
                        node.id,
                    )?;
                    owners.insert(node.id, DagOwnerOrigin::BorrowedLoad);
                    states.insert(node.id, DagOwnerState::BorrowedLive);
                }
                RiscOp::Copy => {
                    require_dag_arity(node.id, "copy", 1, node.inputs.len())?;
                    require_live_dag_owner(&states, node.inputs[0], node.id)?;
                    require_exact_dag_directive(
                        directive,
                        &DagDirective::Clone {
                            node: node.id,
                            source: node.inputs[0],
                        },
                        node.id,
                    )?;
                    owners.insert(node.id, DagOwnerOrigin::OwnedProducer);
                    states.insert(node.id, DagOwnerState::OwnedLive);
                }
                RiscOp::Drop => {
                    require_dag_arity(node.id, "drop", 1, node.inputs.len())?;
                    match states.get(&node.inputs[0]).copied() {
                        Some(DagOwnerState::BorrowedLive) => require_exact_dag_directive(
                            directive,
                            &DagDirective::BorrowedDrop {
                                node: node.id,
                                source: node.inputs[0],
                            },
                            node.id,
                        )?,
                        Some(DagOwnerState::OwnedLive) => {
                            consume_dag_owner(&mut states, node.inputs[0], node.id)?;
                            require_exact_dag_directive(
                                directive,
                                &DagDirective::OwnedDrop {
                                    node: node.id,
                                    source: node.inputs[0],
                                },
                                node.id,
                            )?;
                        }
                        Some(DagOwnerState::OwnedTerminal) => {
                            return Err(OwnershipError::DagDuplicateTerminal {
                                owner: node.inputs[0].0,
                            });
                        }
                        None => {
                            return Err(OwnershipError::DagInput {
                                node: node.id.0,
                                input: node.inputs[0].0,
                            });
                        }
                    }
                }
                RiscOp::Realize => {
                    require_dag_arity(node.id, "realize", 1, node.inputs.len())?;
                    let source_state = require_live_dag_owner(&states, node.inputs[0], node.id)?;
                    match source_state {
                        DagOwnerState::BorrowedLive => require_exact_dag_directive(
                            directive,
                            &DagDirective::CloneProduce {
                                node: node.id,
                                source: node.inputs[0],
                            },
                            node.id,
                        )?,
                        DagOwnerState::OwnedLive
                            if dag_owner_used_after(dag, node.inputs[0], node.id) =>
                        {
                            require_exact_dag_directive(
                                directive,
                                &DagDirective::CloneProduce {
                                    node: node.id,
                                    source: node.inputs[0],
                                },
                                node.id,
                            )?;
                        }
                        DagOwnerState::OwnedLive => {
                            consume_dag_owner(&mut states, node.inputs[0], node.id)?;
                            require_exact_dag_directive(
                                directive,
                                &DagDirective::MoveProduce {
                                    node: node.id,
                                    source: node.inputs[0],
                                },
                                node.id,
                            )?;
                        }
                        DagOwnerState::OwnedTerminal => {
                            unreachable!("require_live_dag_owner rejects terminal owners")
                        }
                    }
                    owners.insert(node.id, DagOwnerOrigin::OwnedProducer);
                    states.insert(node.id, DagOwnerState::OwnedLive);
                }
                RiscOp::Store { .. } => {
                    require_dag_arity(node.id, "store", 1, node.inputs.len())?;
                    let source_state = require_live_dag_owner(&states, node.inputs[0], node.id)?;
                    if source_state == DagOwnerState::BorrowedLive {
                        require_exact_dag_directive(
                            directive,
                            &DagDirective::CloneStore {
                                node: node.id,
                                source: node.inputs[0],
                            },
                            node.id,
                        )?;
                    } else {
                        consume_dag_owner(&mut states, node.inputs[0], node.id)?;
                        require_exact_dag_directive(
                            directive,
                            &DagDirective::MoveStore {
                                node: node.id,
                                source: node.inputs[0],
                            },
                            node.id,
                        )?;
                    }
                }
                _ => {
                    let mut borrows = node.inputs.clone();
                    // A node reads its activation to decide whether it checks.
                    for dependency in node.shape_deps.iter().chain(&node.owner.activation) {
                        if !borrows.contains(dependency) {
                            borrows.push(*dependency);
                        }
                    }
                    for source in &borrows {
                        require_live_dag_owner(&states, *source, node.id)?;
                    }
                    require_exact_dag_directive(
                        directive,
                        &DagDirective::Produce {
                            node: node.id,
                            borrows,
                        },
                        node.id,
                    )?;
                    owners.insert(node.id, DagOwnerOrigin::OwnedProducer);
                    states.insert(node.id, DagOwnerState::OwnedLive);
                }
            }
        }
        if self.owners != owners {
            return Err(dag_directive_error(
                "owner origins diverge from the retained DAG payload".to_string(),
            ));
        }

        for (root_index, root) in dag.roots().iter().enumerate() {
            let directive = self
                .directives
                .get(cursor)
                .ok_or_else(|| dag_directive_error(format!("root n{} has no directive", root.0)))?;
            cursor += 1;
            if matches!(
                dag.get(*root).map(|node| &node.op),
                Some(RiscOp::Store { .. })
            ) {
                require_exact_dag_directive(
                    directive,
                    &DagDirective::StoreRoot { node: *root },
                    *root,
                )?;
                continue;
            }
            let consumer = NodeId(dag.nodes().len() + root_index);
            match states.get(root).copied() {
                Some(DagOwnerState::OwnedLive) => {
                    consume_dag_owner(&mut states, *root, consumer)?;
                    require_exact_dag_directive(
                        directive,
                        &DagDirective::Root { source: *root },
                        consumer,
                    )?;
                }
                Some(DagOwnerState::BorrowedLive) => require_exact_dag_directive(
                    directive,
                    &DagDirective::RootClone { source: *root },
                    consumer,
                )?,
                Some(DagOwnerState::OwnedTerminal) => {
                    return Err(OwnershipError::DagDuplicateTerminal { owner: root.0 });
                }
                None => return Err(OwnershipError::DagInvalidRoot { root: root.0 }),
            }
        }

        let live_owned = states
            .iter()
            .filter_map(|(owner, state)| (*state == DagOwnerState::OwnedLive).then_some(*owner))
            .collect::<Vec<_>>();
        for owner in live_owned {
            let directive = self.directives.get(cursor).ok_or_else(|| {
                dag_directive_error(format!("owned producer n{} has no terminal", owner.0))
            })?;
            cursor += 1;
            let consumer = NodeId(dag.nodes().len() + dag.roots().len());
            consume_dag_owner(&mut states, owner, consumer)?;
            require_exact_dag_directive(
                directive,
                &DagDirective::ScopeDrop { source: owner },
                consumer,
            )?;
        }
        if cursor != self.directives.len() {
            return Err(dag_directive_error(format!(
                "{} directives remain after the payload is exhausted",
                self.directives.len() - cursor
            )));
        }
        for (&owner, origin) in &owners {
            if *origin == DagOwnerOrigin::OwnedProducer
                && states.get(&owner) != Some(&DagOwnerState::OwnedTerminal)
            {
                return Err(OwnershipError::DagMissingTerminal { owner: owner.0 });
            }
        }
        Ok(())
    }

    pub fn render(&self) -> String {
        use core::fmt::Write as _;
        let mut out = String::new();
        for directive in &self.directives {
            match directive {
                DagDirective::BorrowLoad { node } => {
                    let _ = writeln!(out, "borrow load n{}", node.0);
                }
                DagDirective::Produce { node, borrows } => {
                    let inputs = borrows
                        .iter()
                        .map(|input| format!("n{}", input.0))
                        .collect::<Vec<_>>()
                        .join(",");
                    let _ = writeln!(out, "produce n{} borrow [{inputs}]", node.0);
                }
                DagDirective::Clone { node, source } => {
                    let _ = writeln!(out, "clone n{} from n{}", node.0, source.0);
                }
                DagDirective::CloneProduce { node, source } => {
                    let _ = writeln!(out, "produce n{} clone n{}", node.0, source.0);
                }
                DagDirective::MoveProduce { node, source } => {
                    let _ = writeln!(out, "produce n{} move n{}", node.0, source.0);
                }
                DagDirective::CloneStore { node, source } => {
                    let _ = writeln!(out, "store n{} clone n{}", node.0, source.0);
                }
                DagDirective::MoveStore { node, source } => {
                    let _ = writeln!(out, "store n{} move n{}", node.0, source.0);
                }
                DagDirective::StoreRoot { node } => {
                    let _ = writeln!(out, "store-root n{}", node.0);
                }
                DagDirective::BorrowedDrop { node, source } => {
                    let _ = writeln!(out, "drop n{} borrow n{}", node.0, source.0);
                }
                DagDirective::OwnedDrop { node, source } => {
                    let _ = writeln!(out, "drop n{} move n{}", node.0, source.0);
                }
                DagDirective::Root { source } => {
                    let _ = writeln!(out, "root move n{}", source.0);
                }
                DagDirective::RootClone { source } => {
                    let _ = writeln!(out, "root clone n{}", source.0);
                }
                DagDirective::ScopeDrop { source } => {
                    let _ = writeln!(out, "scope-drop move n{}", source.0);
                }
            }
        }
        out
    }
}

fn require_dag_arity(
    node: NodeId,
    operation: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), OwnershipError> {
    if expected == actual {
        Ok(())
    } else {
        Err(OwnershipError::DagArity {
            node: node.0,
            operation,
            expected,
            actual,
        })
    }
}

fn validate_dag_dependencies(
    dag: &Dag,
    node: &chelis_dag_core::dag::DagNode,
) -> Result<(), OwnershipError> {
    for input in node.dependencies() {
        if input.0 >= node.id.0 || dag.get(input).is_none() {
            return Err(OwnershipError::DagInput {
                node: node.id.0,
                input: input.0,
            });
        }
    }
    Ok(())
}

fn dag_owner_used_after(dag: &Dag, owner: NodeId, consumer: NodeId) -> bool {
    dag.nodes().iter().skip(consumer.0 + 1).any(|node| {
        node.inputs.contains(&owner)
            || node.shape_deps.contains(&owner)
            || node.result_claim_deps.contains(&owner)
            || node.owner.activation == Some(owner)
    }) || dag.roots().contains(&owner)
}

fn require_live_dag_owner(
    states: &BTreeMap<NodeId, DagOwnerState>,
    owner: NodeId,
    consumer: NodeId,
) -> Result<DagOwnerState, OwnershipError> {
    match states.get(&owner).copied() {
        Some(DagOwnerState::BorrowedLive) => Ok(DagOwnerState::BorrowedLive),
        Some(DagOwnerState::OwnedLive) => Ok(DagOwnerState::OwnedLive),
        Some(DagOwnerState::OwnedTerminal) => Err(OwnershipError::DagUseAfterTerminal {
            owner: owner.0,
            consumer: consumer.0,
        }),
        None => Err(OwnershipError::DagInput {
            node: consumer.0,
            input: owner.0,
        }),
    }
}

fn consume_dag_owner(
    states: &mut BTreeMap<NodeId, DagOwnerState>,
    owner: NodeId,
    consumer: NodeId,
) -> Result<(), OwnershipError> {
    match states.get(&owner).copied() {
        Some(DagOwnerState::BorrowedLive) => Err(OwnershipError::DagBorrowConsumed {
            owner: owner.0,
            consumer: consumer.0,
        }),
        Some(DagOwnerState::OwnedLive) => {
            states.insert(owner, DagOwnerState::OwnedTerminal);
            Ok(())
        }
        Some(DagOwnerState::OwnedTerminal) => {
            Err(OwnershipError::DagDuplicateTerminal { owner: owner.0 })
        }
        None => Err(OwnershipError::DagInput {
            node: consumer.0,
            input: owner.0,
        }),
    }
}

fn require_exact_dag_directive(
    actual: &DagDirective,
    expected: &DagDirective,
    consumer: NodeId,
) -> Result<(), OwnershipError> {
    if actual == expected {
        Ok(())
    } else {
        Err(dag_directive_error(format!(
            "directive for ownership consumer n{} does not match its payload operation",
            consumer.0
        )))
    }
}

fn dag_directive_error(detail: String) -> OwnershipError {
    OwnershipError::DagDirectiveMap { detail }
}

/// Ordinary graph input for capacity and storage calculations.
#[derive(Clone, Copy)]
pub struct DagAnalysisView<'a> {
    dag: &'a Dag,
    scope: crate::capacity_key::CapacityScope,
}
impl<'a> DagAnalysisView<'a> {
    pub const fn new(dag: &'a Dag, scope: crate::capacity_key::CapacityScope) -> Self {
        Self { dag, scope }
    }
    pub fn get(self, node: NodeId) -> Option<&'a DagNode> {
        self.dag.get(node)
    }
    pub fn nodes(self) -> &'a [DagNode] {
        self.dag.nodes()
    }
    pub fn roots(self) -> &'a [NodeId] {
        self.dag.roots()
    }
    pub fn len(self) -> usize {
        self.dag.len()
    }
    pub fn is_empty(self) -> bool {
        self.dag.is_empty()
    }
    pub const fn capacity_scope(self) -> crate::capacity_key::CapacityScope {
        self.scope
    }
    pub fn output_axis_sources(
        self,
        node: NodeId,
    ) -> Vec<chelis_dag_core::axis_sources::AxisSource> {
        chelis_dag_core::axis_sources::output_axis_sources(self.dag, node)
    }
    pub fn topological_order(self) -> Vec<NodeId> {
        self.dag.topological_order()
    }
    pub fn reduction_inlined_fused_elems(self) -> chelis_unord::UnordSet<NodeId> {
        chelis_dag_core::fuse::reduction_inlined_fused_elems(self.dag)
    }
    pub fn reduction_inlined_products(self) -> Vec<chelis_dag_core::fuse::InlinedProduct> {
        chelis_dag_core::fuse::reduction_inlined_products(self.dag)
    }
    pub fn contraction_views(self) -> crate::contraction_views::ContractionViews {
        let products = self.reduction_inlined_products();
        let specialized = products
            .iter()
            .flat_map(chelis_dag_core::fuse::InlinedProduct::skipped)
            .collect::<Vec<_>>();
        crate::contraction_views::contraction_views_with_products(
            self.dag,
            &[],
            &specialized,
            &products,
        )
    }
}
