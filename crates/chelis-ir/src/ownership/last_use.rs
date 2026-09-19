use std::collections::{BTreeMap, BTreeSet};

use super::classify::{HeapKind, ValueClass};
use super::error::OwnershipError;
use super::ir::{
    ApplyKind, Block, BlockId, EdgeId, EdgeTerminal, HostSiteAction, HostSiteMap, Op, OpId,
    Operation, OperationRole, OwnerId, OwnerOrigin, OwnershipProgram, OwnershipUse, ScheduleState,
    Terminal, Terminator, Unit, UnitId, consuming_container_operand,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum SchedulePoint {
    BlockEntry(BlockId),
    BeforeOperation { block: BlockId, operation: OpId },
    AfterOperation { block: BlockId, operation: OpId },
    BeforeTerminator(BlockId),
    Edge(EdgeId),
    Exit,
}

#[derive(Debug)]
struct ExpandedCfg {
    successors: BTreeMap<SchedulePoint, BTreeSet<SchedulePoint>>,
    #[allow(
        dead_code,
        reason = "retained as a sealed expanded-CFG scheduling fact"
    )]
    predecessors: BTreeMap<SchedulePoint, BTreeSet<SchedulePoint>>,
}

#[derive(Debug)]
#[allow(
    dead_code,
    reason = "the production scheduler returns these sealed CFG facts for structural tests"
)]
pub(super) struct ScheduleFacts {
    cfgs: BTreeMap<UnitId, ExpandedCfg>,
}

impl ScheduleFacts {
    #[cfg(test)]
    pub(super) fn postdominates(
        &self,
        unit: UnitId,
        candidate: SchedulePoint,
        point: SchedulePoint,
    ) -> bool {
        self.cfgs
            .get(&unit)
            .is_some_and(|cfg| point_postdominates(cfg, candidate, point))
    }

    #[cfg(test)]
    pub(super) fn predecessor_count(&self, unit: UnitId, point: SchedulePoint) -> Option<usize> {
        self.cfgs
            .get(&unit)
            .and_then(|cfg| cfg.predecessors.get(&point))
            .map(BTreeSet::len)
    }

    #[cfg(test)]
    pub(super) fn successor_count(&self, unit: UnitId, point: SchedulePoint) -> Option<usize> {
        self.cfgs
            .get(&unit)
            .and_then(|cfg| cfg.successors.get(&point))
            .map(BTreeSet::len)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Placement {
    InBlock { block: BlockId, after: usize },
    OnEdge { source: BlockId, edge: EdgeId },
}

#[derive(Debug, Clone)]
struct Seed {
    owner: OwnerId,
    id: OpId,
    kind: Terminal,
    definition_order: (u32, u32),
}

#[derive(Debug, Clone)]
struct Placed {
    id: OpId,
    kind: Terminal,
    point: Placement,
    definition_order: (u32, u32),
}

pub(super) fn has_schedule(program: &OwnershipProgram) -> bool {
    program
        .units
        .iter()
        .any(|unit| unit.schedule == ScheduleState::CanonicalLastUse)
}

pub(super) fn schedule(
    program: &mut OwnershipProgram,
    sites: &mut HostSiteMap,
) -> Result<ScheduleFacts, OwnershipError> {
    // Verify the conservative Phase-2 graph before removing any provisional
    // terminal.  Cycles are admitted here, but unsupported structural shapes
    // still fail closed in ordinary ownership verification.
    super::verify::verify(program)?;
    let mut pending_cfgs = program
        .units
        .iter()
        .map(|unit| Ok((unit.id, expanded_cfg(unit)?)))
        .collect::<Result<BTreeMap<_, _>, OwnershipError>>()?;
    let mut cfgs = BTreeMap::new();
    let mut placed_by_unit = Vec::new();
    for (unit_index, unit) in program.units.iter_mut().enumerate() {
        let cfg = pending_cfgs
            .remove(&unit.id)
            .ok_or_else(|| invariant(unit, "preflight lost the unit CFG"))?;
        let highest_id = unit
            .blocks
            .iter()
            .flat_map(|block| {
                block.ops.iter().map(|operation| operation.id.0).chain(
                    block
                        .terminator
                        .edges()
                        .flat_map(|edge| &edge.terminals)
                        .map(|terminal| terminal.id.0),
                )
            })
            .max();
        let mut next_id = match highest_id {
            Some(id) => id
                .checked_add(1)
                .ok_or_else(|| invariant(unit, "scheduled operation identity overflow"))?,
            None => 0,
        };
        let mut seeds = remove_provisionals(unit)?;
        seeds.sort_by_key(|seed| (seed.definition_order, seed.id));
        let mut placements = Vec::new();
        for seed in &seeds {
            if !unit
                .owners
                .get(&seed.owner)
                .is_some_and(|info| info.origin == OwnerOrigin::Owned)
            {
                return Err(noncanonical(
                    unit,
                    seed.owner,
                    "scope exit does not name an owned value",
                ));
            }
            if seed.kind != expected_terminal(unit, seed.owner)? {
                return Err(noncanonical(
                    unit,
                    seed.owner,
                    "scope exit has the wrong typed terminal kind",
                ));
            }
            let (block, after) = definition_start(unit, seed.owner)?;
            let mut owner_placements = Vec::new();
            scheduler_plan_from(unit, seed.owner, block, after, &mut owner_placements)?;
            scheduler_expand_block_entry(unit, &mut owner_placements)?;
            owner_placements.sort();
            owner_placements.dedup();
            check_complete_postdominating_frontier(
                unit,
                &cfg,
                seed.owner,
                block,
                after,
                &owner_placements,
            )?;
            for (index, point) in owner_placements.into_iter().enumerate() {
                let id = if index == 0 {
                    seed.id
                } else {
                    let id = OpId(next_id);
                    next_id = next_id
                        .checked_add(1)
                        .ok_or_else(|| invariant(unit, "scheduled operation identity overflow"))?;
                    id
                };
                placements.push(Placed {
                    id,
                    kind: seed.kind,
                    point,
                    definition_order: seed.definition_order,
                });
            }
        }
        apply_placements(unit, &placements)?;
        unit.schedule = ScheduleState::CanonicalLastUse;
        cfgs.insert(unit.id, cfg);
        placed_by_unit.push((unit_index, placements));
    }
    rebuild_host_sites(program, sites, &placed_by_unit)?;
    upgrade_container_last_use_moves(program, sites)?;
    let facts = ScheduleFacts { cfgs };
    super::verify::verify(program)?;
    Ok(facts)
}

/// chelis#2205: consume a container at its scheduled last use.
///
/// Lowering borrows every builtin operand, so a list whose last use is
/// `append` or `concat` was cloned by the runtime and released one operation
/// later. The scheduler has just placed that release: when a `Drop` terminal
/// for the container operand of a consuming builtin (`consuming_container_operand`)
/// sits directly after the application in the same block, the operand is dead
/// after the call. Turn the borrow into a move, drop the terminal, and remove
/// its host-site action, so the emitter sees a moved operand and the
/// consuming entry point takes ownership. The final `verify` re-checks the
/// move: an owner that is still borrowed elsewhere in the operation, or read
/// again later, fails there rather than here.
///
/// Eligibility is deliberately narrow: an `Owned` list-classed owner, the
/// terminal is a scheduled scope exit, and no other operand of the same
/// application names the owner. Retained aliases are not the scheduler's
/// concern: the strong-owner count at run time decides between an in-place
/// push and a clone, which is why a moved operand can never mutate a view
/// someone else still holds.
fn upgrade_container_last_use_moves(
    program: &mut OwnershipProgram,
    sites: &mut HostSiteMap,
) -> Result<usize, OwnershipError> {
    let mut upgraded = 0;
    for (unit_index, unit) in program.units.iter_mut().enumerate() {
        let owners = &unit.owners;
        for block in &mut unit.blocks {
            let mut index = 0;
            while index < block.ops.len() {
                let candidate = match &block.ops[index].kind {
                    Op::Apply {
                        kind: ApplyKind::Intrinsic,
                        label,
                        args,
                        ..
                    } => consuming_container_operand(label).and_then(|position| {
                        let operand = args.get(position)?;
                        let list_owner = owners.get(&operand.owner).is_some_and(|info| {
                            info.origin == OwnerOrigin::Owned
                                && info.class == ValueClass::Heap(HeapKind::List)
                        });
                        let sole_use = args
                            .iter()
                            .enumerate()
                            .all(|(other, arg)| other == position || arg.owner != operand.owner);
                        (operand.use_ == OwnershipUse::Borrow && list_owner && sole_use)
                            .then_some((position, operand.owner))
                    }),
                    _ => None,
                };
                let Some((position, owner)) = candidate else {
                    index += 1;
                    continue;
                };
                // The scheduler places every terminal owed at this point
                // right after the application, in definition order, so the
                // container's `Drop` may sit behind the terminals of the
                // other operands. Only scheduled scope exits may intervene;
                // any semantic operation means the owner is still live.
                let mut terminal = None;
                for (offset, following) in block.ops[index + 1..].iter().enumerate() {
                    if following.role != OperationRole::ScheduledScopeExit {
                        break;
                    }
                    if let Op::Drop { owner: dropped } = &following.kind
                        && dropped.owner == owner
                        && dropped.use_ == OwnershipUse::Move
                    {
                        terminal = Some((index + 1 + offset, following.id));
                        break;
                    }
                }
                let Some((terminal_index, terminal_id)) = terminal else {
                    index += 1;
                    continue;
                };
                if let Op::Apply { args, schema, .. } = &mut block.ops[index].kind {
                    args[position].use_ = OwnershipUse::Move;
                    schema.operands[position] = OwnershipUse::Move;
                }
                block.ops.remove(terminal_index);
                let block_id = block.id;
                for record in &mut sites.records {
                    record.actions.retain(|action| {
                        !matches!(
                            action,
                            HostSiteAction::Operation {
                                unit,
                                block,
                                operation,
                            } if *unit == unit_index && *block == block_id && *operation == terminal_id
                        )
                    });
                }
                upgraded += 1;
                index += 1;
            }
        }
    }
    Ok(upgraded)
}

pub(super) fn verify_canonical(program: &OwnershipProgram) -> Result<(), OwnershipError> {
    for unit in &program.units {
        if unit.schedule != ScheduleState::CanonicalLastUse {
            if let Some((owner, _)) = provisional_in(unit) {
                return Err(OwnershipError::StaleProvisionalTerminal {
                    unit: unit.name.clone(),
                    owner: owner.0,
                });
            }
            continue;
        }
        let cfg = expanded_cfg(unit)?;
        if let Some((owner, _)) = provisional_in(unit) {
            return Err(OwnershipError::StaleProvisionalTerminal {
                unit: unit.name.clone(),
                owner: owner.0,
            });
        }
        for block in &unit.blocks {
            for edge in block.terminator.edges() {
                for terminal in &edge.terminals {
                    if edge.args.iter().any(|arg| {
                        arg.owner == terminal.kind.owner() && arg.use_ == OwnershipUse::Move
                    }) {
                        return Err(OwnershipError::CarriedOwnerDropped {
                            unit: unit.name.clone(),
                            owner: terminal.kind.owner().0,
                            edge: edge.id.0,
                        });
                    }
                }
            }
        }
        let expected_owners = movable_owners(unit);
        let expected_owner_set = expected_owners.iter().copied().collect::<BTreeSet<_>>();
        for (owner, kind) in actual_terminals(unit)? {
            if !expected_owner_set.contains(&owner) {
                return Err(noncanonical(
                    unit,
                    owner,
                    "scheduled terminal does not name a movable owned value",
                ));
            }
            if kind != expected_terminal(unit, owner)? {
                return Err(noncanonical(
                    unit,
                    owner,
                    "scheduled terminal has the wrong typed kind",
                ));
            }
        }
        for owner in expected_owners {
            let (block, after) = definition_start(unit, owner)?;
            let mut expected = Vec::new();
            verifier_plan_from(unit, owner, block, after, &mut expected)?;
            verifier_expand_block_entry(unit, &mut expected)?;
            expected.sort();
            expected.dedup();
            check_complete_postdominating_frontier(unit, &cfg, owner, block, after, &expected)?;
            let mut actual = actual_placements(unit, owner);
            actual.sort();
            if actual != expected {
                return Err(noncanonical(
                    unit,
                    owner,
                    &format!("expected {expected:?}, found {actual:?}"),
                ));
            }
        }
        verify_terminal_order(unit)?;
    }
    Ok(())
}

fn expanded_cfg(unit: &Unit) -> Result<ExpandedCfg, OwnershipError> {
    let mut successors = BTreeMap::<_, BTreeSet<_>>::new();
    successors.entry(SchedulePoint::Exit).or_default();
    for block in &unit.blocks {
        for edge in block.terminator.edges() {
            if !unit
                .blocks
                .iter()
                .any(|candidate| candidate.id == edge.target)
            {
                return Err(OwnershipError::MissingBlock {
                    unit: unit.name.clone(),
                    block: edge.target.0,
                });
            }
        }
        let ops = non_scheduled_ops(block).collect::<Vec<_>>();
        let entry = SchedulePoint::BlockEntry(block.id);
        let first = ops
            .first()
            .map_or(SchedulePoint::BeforeTerminator(block.id), |op| {
                SchedulePoint::BeforeOperation {
                    block: block.id,
                    operation: op.id,
                }
            });
        successors.entry(entry).or_default().insert(first);
        for (index, operation) in ops.iter().enumerate() {
            let before = SchedulePoint::BeforeOperation {
                block: block.id,
                operation: operation.id,
            };
            let point = SchedulePoint::AfterOperation {
                block: block.id,
                operation: operation.id,
            };
            successors.entry(before).or_default().insert(point);
            let next =
                ops.get(index + 1)
                    .map_or(SchedulePoint::BeforeTerminator(block.id), |next| {
                        SchedulePoint::BeforeOperation {
                            block: block.id,
                            operation: next.id,
                        }
                    });
            successors.entry(point).or_default().insert(next);
        }
        let before = SchedulePoint::BeforeTerminator(block.id);
        let edges = block.terminator.edges().collect::<Vec<_>>();
        if edges.is_empty() {
            successors
                .entry(before)
                .or_default()
                .insert(SchedulePoint::Exit);
        } else {
            for edge in edges {
                let edge_point = SchedulePoint::Edge(edge.id);
                successors.entry(before).or_default().insert(edge_point);
                successors
                    .entry(edge_point)
                    .or_default()
                    .insert(SchedulePoint::BlockEntry(edge.target));
            }
        }
    }
    let mut predecessors = successors
        .keys()
        .map(|point| (*point, BTreeSet::new()))
        .collect::<BTreeMap<_, _>>();
    for (&point, next) in &successors {
        for successor in next {
            predecessors.entry(*successor).or_default().insert(point);
        }
    }
    let mut all = BTreeSet::new();
    let mut pending = vec![SchedulePoint::BlockEntry(unit.entry)];
    while let Some(point) = pending.pop() {
        if all.insert(point) {
            pending.extend(successors.get(&point).into_iter().flatten().copied());
        }
    }
    if !all.contains(&SchedulePoint::Exit) {
        return Err(OwnershipError::LastUseSchedulingUnsupported {
            unit: unit.name.clone(),
            feature: "a nonterminating control-flow component with no exit path",
        });
    }
    if all.len() != successors.len() {
        return Err(invariant(
            unit,
            "expanded CFG contains an unreachable scheduling point",
        ));
    }
    Ok(ExpandedCfg {
        successors,
        predecessors,
    })
}

fn point_postdominates(cfg: &ExpandedCfg, candidate: SchedulePoint, point: SchedulePoint) -> bool {
    if !cfg.successors.contains_key(&candidate) || !cfg.successors.contains_key(&point) {
        return false;
    }
    let mut pending = vec![point];
    let mut visited = BTreeSet::new();
    while let Some(current) = pending.pop() {
        if current == candidate || !visited.insert(current) {
            continue;
        }
        if current == SchedulePoint::Exit {
            return false;
        }
        pending.extend(cfg.successors.get(&current).into_iter().flatten().copied());
    }
    true
}

fn remove_provisionals(unit: &mut Unit) -> Result<Vec<Seed>, OwnershipError> {
    let definitions = definition_orders(unit);
    let mut seeds = Vec::new();
    let unit_name = unit.name.clone();
    for block in &mut unit.blocks {
        let mut kept = Vec::with_capacity(block.ops.len());
        for operation in block.ops.drain(..) {
            if operation.role == OperationRole::ProvisionalScopeExit {
                let kind = terminal_from_op(&operation).ok_or_else(|| {
                    OwnershipError::LoweringInvariant {
                        unit: unit_name.clone(),
                        detail: format!(
                            "operation o{} marks a non-terminal as a scope exit",
                            operation.id.0
                        ),
                    }
                })?;
                let owner = kind.owner();
                let definition_order = definitions.get(&owner).copied().ok_or_else(|| {
                    OwnershipError::IncompleteOwner {
                        unit: unit_name.clone(),
                        owner: owner.0,
                        missing: "definition",
                    }
                })?;
                seeds.push(Seed {
                    owner,
                    id: operation.id,
                    kind,
                    definition_order,
                });
            } else {
                kept.push(operation);
            }
        }
        block.ops = kept;
    }
    let mut owners = BTreeSet::new();
    for seed in &seeds {
        if !owners.insert(seed.owner) {
            return Err(noncanonical(
                unit,
                seed.owner,
                "duplicate provisional scope exit",
            ));
        }
    }
    Ok(seeds)
}

fn apply_placements(unit: &mut Unit, placements: &[Placed]) -> Result<(), OwnershipError> {
    let mut block_sites = BTreeMap::<(BlockId, usize), Vec<&Placed>>::new();
    let mut edge_sites = BTreeMap::<EdgeId, Vec<&Placed>>::new();
    for placed in placements {
        match placed.point {
            Placement::InBlock { block, after } => {
                block_sites.entry((block, after)).or_default().push(placed);
            }
            Placement::OnEdge { edge, .. } => edge_sites.entry(edge).or_default().push(placed),
        }
    }
    for values in block_sites.values_mut().chain(edge_sites.values_mut()) {
        values.sort_by_key(|placed| std::cmp::Reverse(placed.definition_order));
    }
    for block in &mut unit.blocks {
        let old = std::mem::take(&mut block.ops);
        let mut rebuilt = Vec::new();
        for index in 0..=old.len() {
            if let Some(terminals) = block_sites.get(&(block.id, index)) {
                rebuilt.extend(terminals.iter().map(|placed| Operation {
                    id: placed.id,
                    role: OperationRole::ScheduledScopeExit,
                    kind: terminal_op(placed.kind),
                }));
            }
            if let Some(operation) = old.get(index) {
                rebuilt.push(operation.clone());
            }
        }
        block.ops = rebuilt;
        for edge in block.terminator.edges_mut() {
            if let Some(terminals) = edge_sites.get(&edge.id) {
                edge.terminals
                    .extend(terminals.iter().map(|placed| EdgeTerminal {
                        id: placed.id,
                        kind: placed.kind,
                    }));
            }
        }
    }
    Ok(())
}

fn scheduler_plan_from(
    unit: &Unit,
    owner: OwnerId,
    block: BlockId,
    start: usize,
    placements: &mut Vec<Placement>,
) -> Result<(), OwnershipError> {
    let future = scheduler_future_need(unit, owner);
    let mut pending = vec![(block, start)];
    let mut visited = BTreeSet::new();
    while let Some((block, start)) = pending.pop() {
        if !visited.insert((block, start)) {
            continue;
        }
        let block_ref = find_block(unit, block)?;
        let ops = non_scheduled_ops(block_ref).collect::<Vec<_>>();
        if ops
            .iter()
            .skip(start)
            .any(|operation| operation_fixed_consume(&operation.kind, owner))
            || terminator_fixed_consume(&block_ref.terminator, owner)
        {
            continue;
        }
        let last_local = ops
            .iter()
            .enumerate()
            .skip(start)
            .filter(|(_, operation)| operation_nonterminal_use(&operation.kind, owner))
            .map(|(index, _)| index + 1)
            .next_back();
        let edges = block_ref.terminator.edges().collect::<Vec<_>>();
        let target_futures = edges
            .iter()
            .map(|edge| future.get(&edge.target).copied().unwrap_or(false))
            .collect::<Vec<_>>();
        let leaves_block = terminator_nonterminal_use(&block_ref.terminator, owner)
            || edges
                .iter()
                .any(|edge| edge_nonterminal_use(edge, owner) || edge_fixed_consume(edge, owner))
            || target_futures.iter().any(|needed| *needed);
        if !leaves_block {
            placements.push(Placement::InBlock {
                block,
                after: last_local.unwrap_or(start),
            });
            continue;
        }
        for (edge, has_future) in edges.into_iter().zip(target_futures) {
            if edge_fixed_consume(edge, owner) {
                continue;
            }
            if has_future {
                pending.push((edge.target, 0));
            } else {
                placements.push(Placement::OnEdge {
                    source: block,
                    edge: edge.id,
                });
            }
        }
    }
    Ok(())
}

/// Deliberately separate from `scheduler_plan_from`: verification recomputes
/// the death frontier from structural uses and never consumes scheduler facts.
fn verifier_plan_from(
    unit: &Unit,
    owner: OwnerId,
    block: BlockId,
    start: usize,
    expected: &mut Vec<Placement>,
) -> Result<(), OwnershipError> {
    let future = verifier_future_need(unit, owner);
    let mut pending = vec![(block, start)];
    let mut visited = BTreeSet::new();
    while let Some((block, start)) = pending.pop() {
        if !visited.insert((block, start)) {
            continue;
        }
        let current = find_block(unit, block)?;
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
            .map(|edge| future.get(&edge.target).copied().unwrap_or(false))
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
    Ok(())
}

fn scheduler_expand_block_entry(
    unit: &Unit,
    placements: &mut Vec<Placement>,
) -> Result<(), OwnershipError> {
    let mut expanded = Vec::new();
    for placement in placements.drain(..) {
        match placement {
            Placement::InBlock { block, after: 0 } if block != unit.entry => {
                let incoming = unit
                    .blocks
                    .iter()
                    .flat_map(|source| {
                        source
                            .terminator
                            .edges()
                            .filter(move |edge| edge.target == block)
                            .map(move |edge| Placement::OnEdge {
                                source: source.id,
                                edge: edge.id,
                            })
                    })
                    .collect::<Vec<_>>();
                if incoming.is_empty() {
                    return Err(invariant(
                        unit,
                        "non-entry block has no incoming scheduling edge",
                    ));
                }
                expanded.extend(incoming);
            }
            other => expanded.push(other),
        }
    }
    *placements = expanded;
    Ok(())
}

/// Independent entry-boundary expansion for canonical verification.
fn verifier_expand_block_entry(
    unit: &Unit,
    expected: &mut Vec<Placement>,
) -> Result<(), OwnershipError> {
    let mut rebuilt = Vec::new();
    for point in expected.drain(..) {
        if let Placement::InBlock { block, after: 0 } = point
            && block != unit.entry
        {
            let mut found = false;
            for source in &unit.blocks {
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
    *expected = rebuilt;
    Ok(())
}

/// Scheduler-owned monotone fixed point. Re-entering a block which defines
/// the same structural owner starts a new loop-iteration instance and cannot
/// keep the preceding instance live.
fn scheduler_future_need(unit: &Unit, owner: OwnerId) -> BTreeMap<BlockId, bool> {
    let mut need = unit
        .blocks
        .iter()
        .map(|block| (block.id, false))
        .collect::<BTreeMap<_, _>>();
    loop {
        let mut changed = false;
        for block in unit.blocks.iter().rev() {
            let value = scheduler_block_need(unit, block, owner, &need);
            if need[&block.id] != value {
                need.insert(block.id, value);
                changed = true;
            }
        }
        if !changed {
            return need;
        }
    }
}

fn scheduler_block_need(
    _unit: &Unit,
    block: &Block,
    owner: OwnerId,
    need: &BTreeMap<BlockId, bool>,
) -> bool {
    if block.params.iter().any(|param| param.owner == owner) {
        return false;
    }
    let mut local = false;
    for operation in non_scheduled_ops(block) {
        if destination(&operation.kind) == Some(owner) {
            return local;
        }
        local |= operation_nonterminal_use(&operation.kind, owner)
            || operation_fixed_consume(&operation.kind, owner);
    }
    local
        || terminator_nonterminal_use(&block.terminator, owner)
        || terminator_fixed_consume(&block.terminator, owner)
        || block.terminator.edges().any(|edge| {
            edge_nonterminal_use(edge, owner)
                || edge_fixed_consume(edge, owner)
                || need.get(&edge.target).copied().unwrap_or(false)
        })
}

/// Canonical verification deliberately owns a second fixed-point traversal;
/// it consumes no scheduler liveness or placement facts.
fn verifier_future_need(unit: &Unit, owner: OwnerId) -> BTreeMap<BlockId, bool> {
    let mut result = unit
        .blocks
        .iter()
        .map(|block| (block.id, false))
        .collect::<BTreeMap<_, _>>();
    let mut changed = true;
    while changed {
        changed = false;
        for current in &unit.blocks {
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
}

fn rebuild_host_sites(
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
            let (site_index, anchor) = match terminal.point {
                Placement::OnEdge { source, edge } => {
                    sites
                        .records
                        .iter()
                        .enumerate()
                        .find_map(|(index, record)| {
                            edge_anchor_position(record, *unit_index, unit, source, edge)
                                .map(|anchor| (index, anchor + 1))
                        })
                }
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
                                        && record.kind == super::ir::HostSiteKind::FunctionEntry)
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
    record: &super::ir::HostSiteRecord,
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

fn actual_placements(unit: &Unit, owner: OwnerId) -> Vec<Placement> {
    let mut result = Vec::new();
    for block in &unit.blocks {
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
    result
}

fn actual_terminals(unit: &Unit) -> Result<Vec<(OwnerId, Terminal)>, OwnershipError> {
    let mut result = Vec::new();
    for block in &unit.blocks {
        for operation in &block.ops {
            if operation.role == OperationRole::ScheduledScopeExit {
                let kind = terminal_kind(unit, operation)?;
                result.push((kind.owner(), kind));
            }
        }
        result.extend(
            block
                .terminator
                .edges()
                .flat_map(|edge| &edge.terminals)
                .map(|terminal| (terminal.kind.owner(), terminal.kind)),
        );
    }
    Ok(result)
}

fn movable_owners(unit: &Unit) -> Vec<OwnerId> {
    unit.owners
        .iter()
        .filter_map(|(&owner, info)| (info.origin == OwnerOrigin::Owned).then_some(owner))
        .collect()
}

fn operation_nonterminal_use(op: &Op, owner: OwnerId) -> bool {
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

fn operation_fixed_consume(op: &Op, owner: OwnerId) -> bool {
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

fn terminator_nonterminal_use(terminator: &Terminator, owner: OwnerId) -> bool {
    match terminator {
        Terminator::Branch { condition, .. } => condition.owner == owner,
        Terminator::Match { scrutinee, .. } => scrutinee.owner == owner,
        Terminator::Loop { list, .. } => list.owner == owner,
        Terminator::Return { .. } | Terminator::Jump(_) | Terminator::Exit => false,
    }
}

fn terminator_fixed_consume(terminator: &Terminator, owner: OwnerId) -> bool {
    matches!(terminator, Terminator::Return { result } if result.owner == owner && result.use_ == OwnershipUse::Move)
}

fn edge_nonterminal_use(edge: &super::ir::Edge, owner: OwnerId) -> bool {
    edge.args
        .iter()
        .any(|arg| arg.owner == owner && arg.use_ != OwnershipUse::Move)
}

fn edge_fixed_consume(edge: &super::ir::Edge, owner: OwnerId) -> bool {
    edge.args
        .iter()
        .any(|arg| arg.owner == owner && arg.use_ == OwnershipUse::Move)
}

fn definition_start(unit: &Unit, owner: OwnerId) -> Result<(BlockId, usize), OwnershipError> {
    for block in &unit.blocks {
        if block.params.iter().any(|param| param.owner == owner) {
            return Ok((block.id, 0));
        }
        let mut non_scheduled = 0;
        for operation in &block.ops {
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

fn definition_orders(unit: &Unit) -> BTreeMap<OwnerId, (u32, u32)> {
    let mut result = BTreeMap::new();
    for block in &unit.blocks {
        for param in &block.params {
            result.insert(param.owner, (block.id.0, 0));
        }
        for operation in &block.ops {
            if let Some(owner) = destination(&operation.kind) {
                result.insert(owner, (block.id.0, operation.id.0.saturating_add(1)));
            }
        }
    }
    result
}

fn destination(op: &Op) -> Option<OwnerId> {
    match op {
        Op::Define { dest, .. } | Op::Copy { dest, .. } | Op::LoopItem { dest, .. } => Some(*dest),
        Op::Apply { dest, .. } => *dest,
        Op::Project { .. } | Op::Drop { .. } | Op::Discard { .. } | Op::RootConsume { .. } => None,
    }
}

fn non_scheduled_ops(block: &Block) -> impl Iterator<Item = &Operation> {
    block.ops.iter().filter(|operation| {
        !matches!(
            operation.role,
            OperationRole::ProvisionalScopeExit | OperationRole::ScheduledScopeExit
        )
    })
}

fn terminal_kind(unit: &Unit, operation: &Operation) -> Result<Terminal, OwnershipError> {
    terminal_from_op(operation).ok_or_else(|| {
        invariant(
            unit,
            &format!(
                "operation o{} marks a non-terminal as a scope exit",
                operation.id.0
            ),
        )
    })
}

fn terminal_from_op(operation: &Operation) -> Option<Terminal> {
    match &operation.kind {
        Op::Drop { owner } => Some(Terminal::Drop(owner.owner)),
        Op::Discard { owner } => Some(Terminal::Discard(*owner)),
        _ => None,
    }
}

fn expected_terminal(unit: &Unit, owner: OwnerId) -> Result<Terminal, OwnershipError> {
    let info = unit
        .owners
        .get(&owner)
        .ok_or_else(|| OwnershipError::IncompleteOwner {
            unit: unit.name.clone(),
            owner: owner.0,
            missing: "metadata",
        })?;
    Ok(if info.class.is_heap() {
        Terminal::Drop(owner)
    } else {
        Terminal::Discard(owner)
    })
}

fn check_complete_postdominating_frontier(
    unit: &Unit,
    cfg: &ExpandedCfg,
    owner: OwnerId,
    definition_block: BlockId,
    definition_after: usize,
    placements: &[Placement],
) -> Result<(), OwnershipError> {
    // A multi-placement result is a path-local death frontier: the scheduler
    // and verifier independently reconstruct the full set from structural
    // future-use facts, and no one member necessarily post-dominates the
    // definition. A single placement has no such collective proof and must
    // itself post-dominate the definition point.
    if definition_after == 0 && definition_block != unit.entry {
        // A block parameter is materialized by each incoming edge. Its unused
        // death point is therefore edge-local, immediately after transfer,
        // rather than a point following the merged BlockEntry node.
        return Ok(());
    }
    let mut frontier = placements
        .iter()
        .copied()
        .map(|placement| placement_point(unit, placement))
        .collect::<Result<BTreeSet<_>, _>>()?;
    frontier.extend(fixed_consume_points(unit, owner));
    if frontier.len() == 1 {
        let definition = placement_point(
            unit,
            Placement::InBlock {
                block: definition_block,
                after: definition_after,
            },
        )?;
        let terminal = *frontier
            .first()
            .expect("single complete terminal frontier has one point");
        require_postdominates(unit, cfg, terminal, definition)?;
    }
    Ok(())
}

fn require_postdominates(
    unit: &Unit,
    cfg: &ExpandedCfg,
    candidate: SchedulePoint,
    point: SchedulePoint,
) -> Result<(), OwnershipError> {
    if point_postdominates(cfg, candidate, point) {
        Ok(())
    } else {
        Err(invariant(
            unit,
            "single death frontier does not post-dominate its definition",
        ))
    }
}

#[cfg(test)]
pub(super) fn require_postdominates_for_test(
    unit: &Unit,
    candidate: SchedulePoint,
    point: SchedulePoint,
) -> Result<(), OwnershipError> {
    let cfg = expanded_cfg(unit)?;
    require_postdominates(unit, &cfg, candidate, point)
}

fn fixed_consume_points(unit: &Unit, owner: OwnerId) -> BTreeSet<SchedulePoint> {
    let mut points = BTreeSet::new();
    for block in &unit.blocks {
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

fn placement_point(unit: &Unit, placement: Placement) -> Result<SchedulePoint, OwnershipError> {
    match placement {
        Placement::OnEdge { edge, .. } => Ok(SchedulePoint::Edge(edge)),
        Placement::InBlock { block, after: 0 } => Ok(SchedulePoint::BlockEntry(block)),
        Placement::InBlock { block, after } => {
            let operation = non_scheduled_ops(find_block(unit, block)?)
                .nth(after - 1)
                .ok_or_else(|| {
                    invariant(unit, "terminal placement has no stable operation anchor")
                })?;
            Ok(SchedulePoint::AfterOperation {
                block,
                operation: operation.id,
            })
        }
    }
}

fn terminal_op(terminal: Terminal) -> Op {
    match terminal {
        Terminal::Drop(owner) => Op::Drop {
            owner: super::ir::Operand::move_(owner),
        },
        Terminal::Discard(owner) => Op::Discard { owner },
    }
}

fn provisional_in(unit: &Unit) -> Option<(OwnerId, OpId)> {
    unit.blocks
        .iter()
        .flat_map(|block| &block.ops)
        .find_map(|operation| {
            (operation.role == OperationRole::ProvisionalScopeExit)
                .then(|| {
                    terminal_kind(unit, operation)
                        .ok()
                        .map(|kind| (kind.owner(), operation.id))
                })
                .flatten()
        })
}

fn verify_terminal_order(unit: &Unit) -> Result<(), OwnershipError> {
    let definitions = definition_orders(unit);
    for block in &unit.blocks {
        let mut group = Vec::new();
        for operation in &block.ops {
            if operation.role == OperationRole::ScheduledScopeExit {
                group.push(terminal_kind(unit, operation)?.owner());
            } else {
                check_order_group(unit, &definitions, &mut group)?;
            }
        }
        check_order_group(unit, &definitions, &mut group)?;
        for edge in block.terminator.edges() {
            let mut owners = edge
                .terminals
                .iter()
                .map(|terminal| terminal.kind.owner())
                .collect::<Vec<_>>();
            check_order_group(unit, &definitions, &mut owners)?;
        }
    }
    Ok(())
}

fn check_order_group(
    unit: &Unit,
    definitions: &BTreeMap<OwnerId, (u32, u32)>,
    owners: &mut Vec<OwnerId>,
) -> Result<(), OwnershipError> {
    if owners.len() < 2 {
        owners.clear();
        return Ok(());
    }
    let mut expected = owners.clone();
    expected.sort_by_key(|owner| std::cmp::Reverse(definitions[owner]));
    if *owners != expected {
        return Err(noncanonical(
            unit,
            owners[0],
            "co-located terminals are not in reverse definition order",
        ));
    }
    owners.clear();
    Ok(())
}

fn find_block(unit: &Unit, id: BlockId) -> Result<&Block, OwnershipError> {
    unit.blocks
        .iter()
        .find(|block| block.id == id)
        .ok_or_else(|| OwnershipError::MissingBlock {
            unit: unit.name.clone(),
            block: id.0,
        })
}

fn invariant(unit: &Unit, detail: &str) -> OwnershipError {
    OwnershipError::LoweringInvariant {
        unit: unit.name.clone(),
        detail: detail.to_string(),
    }
}

fn noncanonical(unit: &Unit, owner: OwnerId, detail: &str) -> OwnershipError {
    OwnershipError::NonCanonicalTerminal {
        unit: unit.name.clone(),
        owner: owner.0,
        detail: detail.to_string(),
    }
}
