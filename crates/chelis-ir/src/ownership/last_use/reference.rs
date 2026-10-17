//! The scan-based owner lookups [`super::index::UnitIndex`] replaced, kept in
//! the test build as the reference every indexed answer is checked against
//! (chelis#3755). Each scans the unit or a block once per owner, which is the
//! quadratic cost the index removes; their visits count only inside
//! [`super::with_reference_scans_counted`].

use std::collections::{BTreeMap, BTreeSet};

use super::super::error::OwnershipError;
use super::super::ir::{Block, EdgeId, HostSiteAction, HostSiteMap, OwnershipProgram};
use super::super::ir::{
    BlockId, Edge, Op, OpId, OperationRole, OwnerId, OwnershipUse, Terminator, Unit,
};
use super::{
    Placed, Placement, SchedulePoint, destination, invariant, non_scheduled_ops, reference_visited,
    terminal_kind,
};

// The per-owner use predicates the scans evaluate. The production passes read
// the same facts from the index, which derives them from one walk over each
// operand list.

pub(super) fn operation_nonterminal_use(op: &Op, owner: OwnerId) -> bool {
    match op {
        Op::Apply { args, .. } => args
            .iter()
            .any(|arg| arg.owner == owner && arg.use_ != OwnershipUse::Move),
        Op::Copy { source, .. } | Op::Project { source } => {
            source.owner == owner && source.use_ != OwnershipUse::Move
        }
        Op::LoopItem { list, .. } => list.owner == owner && list.use_ != OwnershipUse::Move,
        Op::Define { .. } | Op::Drop { .. } | Op::Discard { .. } | Op::RootConsume { .. } => false,
    }
}

pub(super) fn operation_fixed_consume(op: &Op, owner: OwnerId) -> bool {
    match op {
        Op::Apply { args, .. } => args
            .iter()
            .any(|arg| arg.owner == owner && arg.use_ == OwnershipUse::Move),
        Op::Drop { owner: operand } | Op::RootConsume { owner: operand, .. } => {
            operand.owner == owner
        }
        Op::Discard { owner: discarded } => *discarded == owner,
        Op::Copy { source, .. } | Op::Project { source } => {
            source.owner == owner && source.use_ == OwnershipUse::Move
        }
        Op::LoopItem { list, .. } => list.owner == owner && list.use_ == OwnershipUse::Move,
        Op::Define { .. } => false,
    }
}

pub(super) fn terminator_nonterminal_use(terminator: &Terminator, owner: OwnerId) -> bool {
    match terminator {
        Terminator::Branch { condition, .. } => condition.owner == owner,
        Terminator::Match { scrutinee, .. } => scrutinee.owner == owner,
        Terminator::Loop { list, .. } => list.owner == owner,
        Terminator::Return { .. } | Terminator::Jump(_) | Terminator::Exit => false,
    }
}

pub(super) fn terminator_fixed_consume(terminator: &Terminator, owner: OwnerId) -> bool {
    matches!(terminator, Terminator::Return { result } if result.owner == owner && result.use_ == OwnershipUse::Move)
}

pub(super) fn edge_nonterminal_use(edge: &Edge, owner: OwnerId) -> bool {
    edge.args
        .iter()
        .any(|arg| arg.owner == owner && arg.use_ != OwnershipUse::Move)
}

pub(super) fn edge_fixed_consume(edge: &Edge, owner: OwnerId) -> bool {
    edge.args
        .iter()
        .any(|arg| arg.owner == owner && arg.use_ == OwnershipUse::Move)
}

pub(super) fn definition_start(
    unit: &Unit,
    owner: OwnerId,
) -> Result<(BlockId, usize), OwnershipError> {
    for block in &unit.blocks {
        reference_visited(block.params.len());
        if block.params.iter().any(|param| param.owner == owner) {
            return Ok((block.id, 0));
        }
        let mut non_scheduled = 0;
        for operation in &block.ops {
            reference_visited(1);
            if operation.role == OperationRole::ScheduledScopeExit {
                continue;
            }
            non_scheduled += 1;
            if destination(&operation.kind) == Some(owner) {
                return Ok((block.id, non_scheduled));
            }
        }
    }
    Err(OwnershipError::IncompleteOwner {
        unit: unit.name.clone(),
        owner: owner.0,
        missing: "definition",
    })
}

/// The operation a placement `after` operations into `block` sits behind.
pub(super) fn anchor(unit: &Unit, block: BlockId, after: usize) -> Option<OpId> {
    reference_visited(after);
    non_scheduled_ops(find_block(unit, block).ok()?)
        .nth(after.checked_sub(1)?)
        .map(|operation| operation.id)
}

pub(super) fn fixed_consume_points(unit: &Unit, owner: OwnerId) -> BTreeSet<SchedulePoint> {
    let mut points = BTreeSet::new();
    for block in &unit.blocks {
        reference_visited(block.ops.len());
        for operation in non_scheduled_ops(block) {
            if operation_fixed_consume(&operation.kind, owner) {
                points.insert(SchedulePoint::AfterOperation {
                    block: block.id,
                    operation: operation.id,
                });
            }
        }
        if terminator_fixed_consume(&block.terminator, owner) {
            points.insert(SchedulePoint::BeforeTerminator(block.id));
        }
        for edge in block.terminator.edges() {
            if edge_fixed_consume(edge, owner) {
                points.insert(SchedulePoint::Edge(edge.id));
            }
        }
    }
    points
}

/// The blocks from whose entry `owner` is still needed: the dense fixed point
/// over every block of the unit.
pub(super) fn future_need(unit: &Unit, owner: OwnerId) -> BTreeSet<BlockId> {
    let mut result = unit
        .blocks
        .iter()
        .map(|block| (block.id, false))
        .collect::<BTreeMap<_, _>>();
    let mut changed = true;
    while changed {
        changed = false;
        for current in &unit.blocks {
            reference_visited(current.params.len() + current.ops.len());
            let answer = if current.params.iter().any(|param| param.owner == owner) {
                false
            } else {
                let mut before_redefinition = false;
                let mut redefined = false;
                for operation in non_scheduled_ops(current) {
                    if destination(&operation.kind) == Some(owner) {
                        redefined = true;
                        break;
                    }
                    before_redefinition |= operation_nonterminal_use(&operation.kind, owner);
                    before_redefinition |= operation_fixed_consume(&operation.kind, owner);
                }
                before_redefinition
                    || (!redefined
                        && (terminator_nonterminal_use(&current.terminator, owner)
                            || terminator_fixed_consume(&current.terminator, owner)
                            || current.terminator.edges().any(|edge| {
                                edge_nonterminal_use(edge, owner)
                                    || edge_fixed_consume(edge, owner)
                                    || result.get(&edge.target).copied().unwrap_or(false)
                            })))
            };
            if result[&current.id] != answer {
                result.insert(current.id, answer);
                changed = true;
            }
        }
    }
    result
        .into_iter()
        .filter_map(|(block, needed)| needed.then_some(block))
        .collect()
}

/// The complete sorted death frontier of `owner`, recomputed from its
/// definition by whole-block scans.
pub(super) fn placements(unit: &Unit, owner: OwnerId) -> Result<Vec<Placement>, OwnershipError> {
    let (block, start) = definition_start(unit, owner)?;
    let future = future_need(unit, owner);
    let mut expected = Vec::new();
    let mut pending = vec![(block, start)];
    let mut visited = BTreeSet::new();
    while let Some((block, start)) = pending.pop() {
        if !visited.insert((block, start)) {
            continue;
        }
        let current = find_block(unit, block)?;
        reference_visited(current.ops.len() + current.params.len());
        let operations = non_scheduled_ops(current).collect::<Vec<_>>();
        if operations
            .iter()
            .skip(start)
            .any(|operation| operation_fixed_consume(&operation.kind, owner))
            || terminator_fixed_consume(&current.terminator, owner)
        {
            continue;
        }
        let last_use = operations
            .iter()
            .enumerate()
            .skip(start)
            .rev()
            .find(|(_, operation)| operation_nonterminal_use(&operation.kind, owner))
            .map(|(index, _)| index + 1);
        let successors = current.terminator.edges().collect::<Vec<_>>();
        let successor_needs = successors
            .iter()
            .map(|edge| future.contains(&edge.target))
            .collect::<Vec<_>>();
        let exits_block = terminator_nonterminal_use(&current.terminator, owner)
            || successors
                .iter()
                .any(|edge| edge_nonterminal_use(edge, owner) || edge_fixed_consume(edge, owner))
            || successor_needs.iter().any(|needed| *needed);
        if !exits_block {
            expected.push(Placement::InBlock {
                block,
                after: last_use.unwrap_or(start),
            });
            continue;
        }
        for (edge, needed) in successors.into_iter().zip(successor_needs) {
            if edge_fixed_consume(edge, owner) {
                continue;
            }
            if needed {
                pending.push((edge.target, 0));
            } else {
                expected.push(Placement::OnEdge {
                    source: block,
                    edge: edge.id,
                });
            }
        }
    }
    let mut rebuilt = Vec::new();
    for point in expected {
        if let Placement::InBlock { block, after: 0 } = point
            && block != unit.entry
        {
            let mut found = false;
            for source in &unit.blocks {
                reference_visited(1);
                for edge in source
                    .terminator
                    .edges()
                    .filter(|edge| edge.target == block)
                {
                    found = true;
                    rebuilt.push(Placement::OnEdge {
                        source: source.id,
                        edge: edge.id,
                    });
                }
            }
            if !found {
                return Err(invariant(
                    unit,
                    "non-entry block has no incoming scheduling edge",
                ));
            }
        } else {
            rebuilt.push(point);
        }
    }
    rebuilt.sort();
    rebuilt.dedup();
    Ok(rebuilt)
}

/// The scheduled terminals of `owner` as they stand in the unit, sorted.
pub(super) fn actual_placements(unit: &Unit, owner: OwnerId) -> Vec<Placement> {
    let mut result = Vec::new();
    for block in &unit.blocks {
        reference_visited(block.ops.len());
        let mut non_scheduled = 0;
        for operation in &block.ops {
            if operation.role == OperationRole::ScheduledScopeExit {
                if terminal_kind(unit, operation).is_ok_and(|kind| kind.owner() == owner) {
                    result.push(Placement::InBlock {
                        block: block.id,
                        after: non_scheduled,
                    });
                }
            } else {
                non_scheduled += 1;
            }
        }
        for edge in block.terminator.edges() {
            result.extend(
                edge.terminals
                    .iter()
                    .filter(|terminal| terminal.kind.owner() == owner)
                    .map(|_| Placement::OnEdge {
                        source: block.id,
                        edge: edge.id,
                    }),
            );
        }
    }
    result.sort();
    result
}

/// The sequential host-site rebuild: every terminal finds its anchor by
/// scanning the site map as earlier insertions left it.
pub(super) fn rebuild_host_sites(
    program: &OwnershipProgram,
    sites: &mut HostSiteMap,
    placed: &[(usize, Vec<Placed>)],
) -> Result<(), OwnershipError> {
    if sites.records.is_empty() {
        return Ok(());
    }
    let ids = placed
        .iter()
        .flat_map(|(unit, terminals)| terminals.iter().map(move |terminal| (*unit, terminal.id)))
        .collect::<BTreeSet<_>>();
    for record in &mut sites.records {
        reference_visited(record.actions.len());
        record.actions.retain(|action| {
            !matches!(
                action,
                HostSiteAction::Operation {
                    unit,
                    operation,
                    ..
                } if ids.contains(&(*unit, *operation))
            )
        });
    }
    let mut inserted = BTreeMap::<(usize, Placement), usize>::new();
    for (unit_index, terminals) in placed {
        let unit = &program.units[*unit_index];
        let mut terminals = terminals.iter().collect::<Vec<_>>();
        terminals
            .sort_by_key(|terminal| (terminal.point, std::cmp::Reverse(terminal.definition_order)));
        for terminal in terminals {
            let (site_index, anchor) =
                match terminal.point {
                    Placement::OnEdge { source, edge } => sites
                        .records
                        .iter()
                        .enumerate()
                        .find_map(|(index, record)| {
                            reference_visited(1 + record.actions.len());
                            edge_anchor_position(record, *unit_index, unit, source, edge)
                                .map(|anchor| (index, anchor + 1))
                        }),
                    Placement::InBlock { block, after } => {
                        let anchor = if after == 0 {
                            None
                        } else {
                            Some(
                                non_scheduled_ops(find_block(unit, block)?)
                                    .nth(after - 1)
                                    .map(|operation| operation.id)
                                    .ok_or_else(|| OwnershipError::HostSiteMap {
                                        detail: format!(
                                            "scheduled terminal o{} has an invalid block anchor",
                                            terminal.id.0
                                        ),
                                    })?,
                            )
                        };
                        sites
                            .records
                            .iter()
                            .enumerate()
                            .find_map(|(index, record)| {
                                reference_visited(1 + record.actions.len());
                                record
                                    .actions
                                    .iter()
                                    .position(|action| match *action {
                                        HostSiteAction::Operation {
                                            unit,
                                            block: action_block,
                                            operation,
                                        } => {
                                            unit == *unit_index
                                                && action_block == block
                                                && anchor == Some(operation)
                                        }
                                        _ => false,
                                    })
                                    .map(|anchor| (index, anchor + 1))
                                    .or_else(|| {
                                        (after == 0
                                            && record.unit == *unit_index
                                            && record.kind
                                                == super::super::ir::HostSiteKind::FunctionEntry)
                                            .then_some((index, 0))
                                    })
                            })
                    }
                }
                .ok_or_else(|| OwnershipError::HostSiteMap {
                    detail: format!(
                        "scheduled terminal o{} has no structural site",
                        terminal.id.0
                    ),
                })?;
            let offset = inserted.entry((site_index, terminal.point)).or_default();
            let block = match terminal.point {
                Placement::InBlock { block, .. } | Placement::OnEdge { source: block, .. } => block,
            };
            sites.records[site_index].actions.insert(
                anchor + *offset,
                HostSiteAction::Operation {
                    unit: *unit_index,
                    block,
                    operation: terminal.id,
                },
            );
            *offset += 1;
        }
    }
    Ok(())
}

fn edge_anchor_position(
    record: &super::super::ir::HostSiteRecord,
    unit_index: usize,
    unit: &Unit,
    source: BlockId,
    edge: EdgeId,
) -> Option<usize> {
    record.actions.iter().position(|action| match *action {
        HostSiteAction::ControlEdge {
            unit,
            edge: action_edge,
            source: action_source,
            ..
        } => unit == unit_index && action_edge == edge && action_source == source,
        HostSiteAction::Terminator {
            unit: action_unit,
            block,
        } if action_unit == unit_index && block == source => unit
            .blocks
            .iter()
            .find(|candidate| candidate.id == source)
            .is_some_and(
                |block| matches!(&block.terminator, Terminator::Jump(jump) if jump.id == edge),
            ),
        _ => false,
    })
}

pub(super) fn find_block(unit: &Unit, id: BlockId) -> Result<&Block, OwnershipError> {
    unit.blocks
        .iter()
        .find(|block| block.id == id)
        .ok_or_else(|| OwnershipError::MissingBlock {
            unit: unit.name.clone(),
            block: id.0,
        })
}

/// Every path from `point` to `Exit` passes through `candidate`: a search
/// from `point` that stops at `candidate` and fails on reaching `Exit`.
pub(super) fn point_postdominates(
    successors: &BTreeMap<SchedulePoint, BTreeSet<SchedulePoint>>,
    candidate: SchedulePoint,
    point: SchedulePoint,
) -> bool {
    if !successors.contains_key(&candidate) || !successors.contains_key(&point) {
        return false;
    }
    let mut pending = vec![point];
    let mut visited = BTreeSet::new();
    while let Some(current) = pending.pop() {
        if current == candidate || !visited.insert(current) {
            continue;
        }
        reference_visited(1);
        if current == SchedulePoint::Exit {
            return false;
        }
        pending.extend(successors.get(&current).into_iter().flatten().copied());
    }
    true
}
