use std::collections::{BTreeMap, BTreeSet};

use super::error::OwnershipError;
use super::ir::{
    ApplyKind, Block, BlockId, EdgeId, EdgeTerminal, HostSiteAction, HostSiteMap, Op, OpId,
    Operand, Operation, OperationRole, OwnerId, OwnerOrigin, OwnershipProgram, OwnershipUse,
    ScheduleState, Terminal, Terminator, Unit, UnitId, container_consumer_operand,
};

mod index;
mod postdominators;
#[cfg(test)]
mod reference;

use index::UnitIndex;
use postdominators::PostDominators;

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
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the post-dominator tree answers production queries; the test build \
                      checks each answer against a search of these successors"
        )
    )]
    successors: BTreeMap<SchedulePoint, BTreeSet<SchedulePoint>>,
    #[allow(
        dead_code,
        reason = "retained as a sealed expanded-CFG scheduling fact"
    )]
    predecessors: BTreeMap<SchedulePoint, BTreeSet<SchedulePoint>>,
    postdominators: PostDominators,
}

#[derive(Debug)]
#[allow(
    dead_code,
    reason = "the production scheduler returns these sealed CFG facts for structural tests"
)]
pub(super) struct ScheduleFacts {
    cfgs: BTreeMap<UnitId, ExpandedCfg>,
    verification: super::verify::HostVerification,
}

impl ScheduleFacts {
    /// The verification of the scheduled program `schedule` returned with.
    /// The program it describes is the one `schedule` left in place; the
    /// caller seals both together (chelis#3730).
    pub(super) fn into_verification(self) -> super::verify::HostVerification {
        self.verification
    }

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

#[cfg(test)]
thread_local! {
    static LOOKUP_VISITS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static REFERENCE_SCANS_COUNTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Operations, parameters, edges, site actions and index entries the
/// per-owner and per-terminal lookups of the scheduler and its canonical
/// re-check examined on this thread.
#[cfg(test)]
pub(super) fn lookup_visits() -> u64 {
    LOOKUP_VISITS.with(std::cell::Cell::get)
}

/// Count the reference scans every indexed lookup is checked against as
/// lookup visits while `run` executes: the negative control that a per-owner
/// or per-terminal scan is visible to [`lookup_visits`].
#[cfg(test)]
pub(super) fn with_reference_scans_counted<T>(run: impl FnOnce() -> T) -> T {
    REFERENCE_SCANS_COUNTED.with(|flag| flag.set(true));
    let result = run();
    REFERENCE_SCANS_COUNTED.with(|flag| flag.set(false));
    result
}

#[cfg(test)]
fn lookup_visited(entries: usize) {
    LOOKUP_VISITS.with(|cell| cell.set(cell.get() + entries as u64));
}

#[cfg(not(test))]
fn lookup_visited(_entries: usize) {}

#[cfg(test)]
fn reference_visited(entries: usize) {
    if REFERENCE_SCANS_COUNTED.with(std::cell::Cell::get) {
        lookup_visited(entries);
    }
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
        let index = UnitIndex::new(unit)?;
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
            let (block, after) = index.definition(seed.owner)?;
            let mut owner_placements = Vec::new();
            scheduler_plan_from(&index, seed.owner, block, after, &mut owner_placements)?;
            scheduler_expand_block_entry(&index, unit, &mut owner_placements)?;
            owner_placements.sort();
            owner_placements.dedup();
            #[cfg(test)]
            assert_eq!(
                Ok(&owner_placements),
                reference::placements(unit, seed.owner).as_ref(),
                "indexed scheduler frontier diverged from the reference scans"
            );
            check_complete_postdominating_frontier(
                &index,
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
    upgrade_copy_last_use_moves(program, sites)?;
    upgrade_container_last_use_moves(program, sites)?;
    let verification = super::verify::verify(program)?;
    Ok(ScheduleFacts { cfgs, verification })
}

/// chelis#2205: consume a container at its scheduled last use.
///
/// Lowering borrows every builtin operand, so a container whose last use is
/// `append`, `concat`, `dict_insert`, `dict_merge` or `dict_remove` was
/// cloned by the runtime and released one operation later. The scheduler has
/// just placed that release: when a `Drop` terminal for the consumed operand
/// of a container-producing builtin (a row of `CONTAINER_CONSUMERS`, matched
/// by label, by the row's heap kind on the result, and by that same kind on
/// the named operand) sits directly after the application in the same block,
/// the operand is dead after the call. Turn the borrow into a move, drop the
/// terminal, and remove its host-site action, so the emitter sees a moved
/// operand and the consuming entry point takes ownership. The final `verify`
/// re-checks the move: an owner that is still borrowed elsewhere in the
/// operation, or read again later, fails there rather than here.
///
/// Eligibility is deliberately narrow: an `Owned` owner carrying the row's
/// heap kind, the terminal is a scheduled scope exit, and no other operand of
/// the same application names the owner. Retained aliases are not the
/// scheduler's concern: the strong-owner count at run time decides between an
/// in-place mutation and a clone, which is why a moved operand can never
/// mutate a view someone else still holds.
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
                        schema,
                        ..
                    } => container_consumer_operand(label, schema.result, |position| {
                        args.get(position)
                            .and_then(|arg| owners.get(&arg.owner))
                            .map(|info| info.class)
                    })
                    .and_then(|position| {
                        let operand = args.get(position)?;
                        let owned = owners
                            .get(&operand.owner)
                            .is_some_and(|info| info.origin == OwnerOrigin::Owned);
                        let sole_use = args
                            .iter()
                            .enumerate()
                            .all(|(other, arg)| other == position || arg.owner != operand.owner);
                        (operand.use_ == OwnershipUse::Borrow && owned && sole_use)
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

/// chelis#2206, chelis#3716: move a named owner at its scheduled last use.
///
/// Lowering hands every call argument a `Copy` of a named owner, because it
/// cannot tell a last use from an earlier one and one call may name the
/// owner in several consuming slots (spec/04's ordinary consuming fan-out).
/// The scheduler can tell: when an owner's scheduled `Drop` follows a `Copy`
/// of it, with only scheduled scope exits in between, that copy was the
/// owner's last read on this path, and the owner dies right after paying for
/// it. Hand the owner itself to the copy's consumer instead: the consumer's
/// operand names the owner, the `Copy` and the `Drop` disappear together with
/// their host-site actions, and the copy's identity leaves the unit. A retain
/// followed by a release becomes nothing.
///
/// One call naming the owner twice gets one copy and one move: lowering
/// emits a copy per slot, in slot order, and only the last copy sits directly
/// before the owner's `Drop`. Which slot moves is therefore decided by the
/// schedule, never by the slot.
///
/// Eligibility is deliberately narrow, so this rewrite can only reproduce the
/// shape a move at that position would have had: the consumer is the copy's
/// one and only use, it moves the copy, it sits later in the same block (an
/// application or the block's `Return` or `Jump`), and it does not already
/// name the owner. The owner then lives from the copy to that consumer and is
/// read by nothing in between, because the scheduler placed its terminal
/// before the consumer. The final `verify` re-checks the result.
///
/// The cost is linear in the program apart from the in-block removals: the
/// operand census is taken once per unit and kept current as rewrites land,
/// and the host-site actions of every removed operation are dropped in one
/// pass over the site map at the end.
fn upgrade_copy_last_use_moves(
    program: &mut OwnershipProgram,
    sites: &mut HostSiteMap,
) -> Result<usize, OwnershipError> {
    let mut upgraded = 0;
    let mut removed = BTreeSet::new();
    for (unit_index, unit) in program.units.iter_mut().enumerate() {
        let mut census = owner_operand_counts(unit);
        let mut block_index = 0;
        while block_index < unit.blocks.len() {
            let mut index = 0;
            while index < unit.blocks[block_index].ops.len() {
                let Some(rewrite) = copy_upgrade_at(unit, &census, block_index, index) else {
                    index += 1;
                    continue;
                };
                apply_copy_upgrade(unit, block_index, &rewrite, &mut census);
                let block = unit.blocks[block_index].id;
                removed.insert((unit_index, block, rewrite.copy_id));
                removed.insert((unit_index, block, rewrite.drop_id));
                upgraded += 1;
                // The `Copy` before this index is gone; the operation that
                // followed the removed `Drop` now sits at `index - 1`.
                index = rewrite.copy_index;
            }
            block_index += 1;
        }
    }
    if !removed.is_empty() {
        for record in &mut sites.records {
            record.actions.retain(|action| match action {
                HostSiteAction::Operation {
                    unit,
                    block,
                    operation,
                } => !removed.contains(&(*unit, *block, *operation)),
                _ => true,
            });
        }
    }
    Ok(upgraded)
}

/// A copy-at-last-use rewrite found by [`copy_upgrade_at`].
struct CopyUpgrade {
    owner: OwnerId,
    copy: OwnerId,
    copy_index: usize,
    copy_id: OpId,
    drop_index: usize,
    drop_id: OpId,
    consumer: CopyConsumer,
}

enum CopyConsumer {
    Operation { index: usize, operand: usize },
    Return,
    Jump { operand: usize },
}

/// Recognize a scheduled `Drop` at `drop_index` that retires the source of
/// the `Copy` directly before it (only scheduled scope exits between).
fn copy_upgrade_at(
    unit: &Unit,
    census: &BTreeMap<OwnerId, usize>,
    block_index: usize,
    drop_index: usize,
) -> Option<CopyUpgrade> {
    let block = &unit.blocks[block_index];
    let terminal = &block.ops[drop_index];
    let Op::Drop { owner: dropped } = &terminal.kind else {
        return None;
    };
    if terminal.role != OperationRole::ScheduledScopeExit || dropped.use_ != OwnershipUse::Move {
        return None;
    }
    let owner = dropped.owner;
    let copy_index = block.ops[..drop_index]
        .iter()
        .rposition(|operation| operation.role != OperationRole::ScheduledScopeExit)?;
    let Op::Copy { dest, source } = &block.ops[copy_index].kind else {
        return None;
    };
    if source.owner != owner || source.use_ != OwnershipUse::Clone {
        return None;
    }
    let copy = *dest;
    if census.get(&copy).copied() != Some(1) {
        return None;
    }
    let names_owner = |operands: &[Operand]| operands.iter().any(|arg| arg.owner == owner);
    let moves_copy = |operands: &[Operand]| {
        operands
            .iter()
            .position(|arg| arg.owner == copy && arg.use_ == OwnershipUse::Move)
    };
    let consumer =
        block.ops[drop_index + 1..]
            .iter()
            .enumerate()
            .find_map(|(offset, operation)| match &operation.kind {
                Op::Apply { args, .. } => moves_copy(args).map(|operand| {
                    (!names_owner(args)).then_some(CopyConsumer::Operation {
                        index: drop_index + 1 + offset,
                        operand,
                    })
                }),
                _ => None,
            });
    let consumer = match consumer {
        Some(found) => found?,
        None => match &block.terminator {
            Terminator::Return { result }
                if result.owner == copy && result.use_ == OwnershipUse::Move =>
            {
                CopyConsumer::Return
            }
            Terminator::Jump(edge) if !names_owner(&edge.args) => CopyConsumer::Jump {
                operand: moves_copy(&edge.args)?,
            },
            _ => return None,
        },
    };
    Some(CopyUpgrade {
        owner,
        copy,
        copy_index,
        copy_id: block.ops[copy_index].id,
        drop_index,
        drop_id: terminal.id,
        consumer,
    })
}

/// Rewrite one recognized copy and keep the operand census current: the copy
/// identity leaves the unit, and the owner loses the copy's source operand and
/// its `Drop` and gains the consumer's operand.
fn apply_copy_upgrade(
    unit: &mut Unit,
    block_index: usize,
    rewrite: &CopyUpgrade,
    census: &mut BTreeMap<OwnerId, usize>,
) {
    let block = &mut unit.blocks[block_index];
    match rewrite.consumer {
        CopyConsumer::Operation { index, operand } => {
            if let Op::Apply { args, .. } = &mut block.ops[index].kind {
                args[operand].owner = rewrite.owner;
            }
        }
        CopyConsumer::Return => {
            if let Terminator::Return { result } = &mut block.terminator {
                result.owner = rewrite.owner;
            }
        }
        CopyConsumer::Jump { operand } => {
            if let Terminator::Jump(edge) = &mut block.terminator {
                edge.args[operand].owner = rewrite.owner;
            }
        }
    }
    block.ops.remove(rewrite.drop_index);
    block.ops.remove(rewrite.copy_index);
    unit.owners.remove(&rewrite.copy);
    census.remove(&rewrite.copy);
    // The owner had at least the copy's source operand and its `Drop`, so
    // the net loss of one cannot underflow.
    if let Some(count) = census.get_mut(&rewrite.owner) {
        *count -= 1;
    }
}

#[cfg(test)]
thread_local! {
    /// How many whole-unit operand censuses the copy rewrite has taken on
    /// this thread, for the test that bounds them at one per unit.
    pub(super) static OPERAND_CENSUSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// For every owner `unit` names, how many operand positions name it:
/// operation operands, terminator operands, edge arguments and edge
/// terminals.
fn owner_operand_counts(unit: &Unit) -> BTreeMap<OwnerId, usize> {
    #[cfg(test)]
    OPERAND_CENSUSES.with(|scans| scans.set(scans.get() + 1));
    let mut counts = BTreeMap::new();
    let mut note = |owner: OwnerId| *counts.entry(owner).or_insert(0) += 1;
    for block in &unit.blocks {
        for operation in &block.ops {
            match &operation.kind {
                Op::Apply { args, .. } => args.iter().for_each(|arg| note(arg.owner)),
                Op::Copy {
                    source: operand, ..
                }
                | Op::Project { source: operand }
                | Op::LoopItem { list: operand, .. }
                | Op::Drop { owner: operand }
                | Op::RootConsume { owner: operand, .. } => note(operand.owner),
                Op::Discard { owner } => note(*owner),
                Op::Define { .. } => {}
            }
        }
        match &block.terminator {
            Terminator::Return { result: operand }
            | Terminator::Branch {
                condition: operand, ..
            }
            | Terminator::Match {
                scrutinee: operand, ..
            }
            | Terminator::Loop { list: operand, .. } => note(operand.owner),
            Terminator::Jump(_) | Terminator::Exit => {}
        }
        for edge in block.terminator.edges() {
            edge.args.iter().for_each(|arg| note(arg.owner));
            edge.terminals
                .iter()
                .for_each(|terminal| note(terminal.kind.owner()));
        }
    }
    counts
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
        let index = UnitIndex::new(unit)?;
        let mut actual_by_owner = actual_placements(unit);
        for owner in expected_owners {
            let (block, after) = index.definition(owner)?;
            let mut expected = Vec::new();
            verifier_plan_from(&index, owner, block, after, &mut expected)?;
            verifier_expand_block_entry(&index, unit, &mut expected)?;
            expected.sort();
            expected.dedup();
            #[cfg(test)]
            assert_eq!(
                Ok(&expected),
                reference::placements(unit, owner).as_ref(),
                "indexed canonical frontier diverged from the reference scans"
            );
            check_complete_postdominating_frontier(
                &index, unit, &cfg, owner, block, after, &expected,
            )?;
            // An owner every path consumes by a fixed move has no scheduled
            // terminal, and its expected frontier is empty too.
            let mut actual = actual_by_owner
                .remove(&owner)
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            actual.sort();
            #[cfg(test)]
            assert_eq!(
                actual,
                reference::actual_placements(unit, owner),
                "grouped scheduled terminals diverged from the reference scan"
            );
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
    let block_ids = unit
        .blocks
        .iter()
        .map(|block| block.id)
        .collect::<BTreeSet<_>>();
    for block in &unit.blocks {
        for edge in block.terminator.edges() {
            if !block_ids.contains(&edge.target) {
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
    let postdominators = PostDominators::new(&successors, &predecessors);
    Ok(ExpandedCfg {
        successors,
        predecessors,
        postdominators,
    })
}

fn point_postdominates(cfg: &ExpandedCfg, candidate: SchedulePoint, point: SchedulePoint) -> bool {
    let answer = cfg.postdominators.postdominates(candidate, point);
    #[cfg(test)]
    assert_eq!(
        answer,
        reference::point_postdominates(&cfg.successors, candidate, point),
        "post-dominator tree diverged from the path search"
    );
    answer
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
    index: &UnitIndex<'_>,
    owner: OwnerId,
    block: BlockId,
    start: usize,
    placements: &mut Vec<Placement>,
) -> Result<(), OwnershipError> {
    let future = scheduler_future_need(index, owner);
    let mut pending = vec![(block, start)];
    let mut visited = BTreeSet::new();
    while let Some((block, start)) = pending.pop() {
        if !visited.insert((block, start)) {
            continue;
        }
        let (position, block_ref) = index.block(block)?;
        let mentions = index.mentions(owner, position);
        if mentions.fixed_consume_from(start) || mentions.terminator_fixed {
            continue;
        }
        let last_local = mentions.last_nonterminal_use_from(start);
        let edges = block_ref.terminator.edges().collect::<Vec<_>>();
        lookup_visited(edges.len());
        let target_futures = edges
            .iter()
            .map(|edge| future.contains(&edge.target))
            .collect::<Vec<_>>();
        let leaves_block = mentions.terminator_nonterminal
            || mentions.edge_use()
            || target_futures.iter().any(|needed| *needed);
        if !leaves_block {
            placements.push(Placement::InBlock {
                block,
                after: last_local.unwrap_or(start),
            });
            continue;
        }
        for (edge_index, (edge, has_future)) in edges.into_iter().zip(target_futures).enumerate() {
            if mentions.edge_fixed_consume(edge_index) {
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
    index: &UnitIndex<'_>,
    owner: OwnerId,
    block: BlockId,
    start: usize,
    expected: &mut Vec<Placement>,
) -> Result<(), OwnershipError> {
    let future = verifier_future_need(index, owner);
    let mut pending = vec![(block, start)];
    let mut visited = BTreeSet::new();
    while let Some((block, start)) = pending.pop() {
        if !visited.insert((block, start)) {
            continue;
        }
        let (position, current) = index.block(block)?;
        let uses = index.mentions(owner, position);
        if uses.fixed_consume_from(start) || uses.terminator_fixed {
            continue;
        }
        let last_use = uses.last_nonterminal_use_from(start);
        let successors = current.terminator.edges().collect::<Vec<_>>();
        lookup_visited(successors.len());
        let successor_needs = successors
            .iter()
            .map(|edge| future.contains(&edge.target))
            .collect::<Vec<_>>();
        let exits_block = uses.terminator_nonterminal
            || uses.edge_use()
            || successor_needs.iter().any(|needed| *needed);
        if !exits_block {
            expected.push(Placement::InBlock {
                block,
                after: last_use.unwrap_or(start),
            });
            continue;
        }
        for (edge_index, (edge, needed)) in successors.into_iter().zip(successor_needs).enumerate()
        {
            if uses.edge_fixed_consume(edge_index) {
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
    index: &UnitIndex<'_>,
    unit: &Unit,
    placements: &mut Vec<Placement>,
) -> Result<(), OwnershipError> {
    let mut expanded = Vec::new();
    for placement in placements.drain(..) {
        match placement {
            Placement::InBlock { block, after: 0 } if block != unit.entry => {
                let incoming = index
                    .incoming(block)
                    .iter()
                    .map(|incoming| Placement::OnEdge {
                        source: incoming.source,
                        edge: incoming.edge,
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
    index: &UnitIndex<'_>,
    unit: &Unit,
    expected: &mut Vec<Placement>,
) -> Result<(), OwnershipError> {
    let mut rebuilt = Vec::new();
    for point in expected.drain(..) {
        if let Placement::InBlock { block, after: 0 } = point
            && block != unit.entry
        {
            let mut found = false;
            for incoming in index.incoming(block) {
                found = true;
                rebuilt.push(Placement::OnEdge {
                    source: incoming.source,
                    edge: incoming.edge,
                });
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

/// Scheduler-owned least fixed point of the blocks from whose entry `owner`
/// is still needed. Re-entering a block which defines the same structural
/// owner starts a new loop-iteration instance and cannot keep the preceding
/// instance live. A block needs the owner on its own account when it is not
/// the owner's parameter block and uses it before redefining it, or never
/// redefines it and uses it in its terminator or on an edge; the need then
/// flows back over every incoming edge from a block that neither binds nor
/// redefines the owner. Only the blocks naming the owner can need it on their
/// own account, so the search starts there and visits only blocks that end up
/// needing it.
fn scheduler_future_need(index: &UnitIndex<'_>, owner: OwnerId) -> BTreeSet<BlockId> {
    let mut need = BTreeSet::new();
    let mut pending = Vec::new();
    for (block, mentions) in index.mentioning_blocks(owner) {
        let local = !mentions.param
            && (mentions.used_before_redefinition()
                || (!mentions.redefines
                    && (mentions.terminator_nonterminal
                        || mentions.terminator_fixed
                        || mentions.edge_use())));
        if local && need.insert(block) {
            pending.push(block);
        }
    }
    while let Some(target) = pending.pop() {
        for incoming in index.incoming(target) {
            let mentions = index.mentions(owner, incoming.source_position);
            if !mentions.param && !mentions.redefines && need.insert(incoming.source) {
                pending.push(incoming.source);
            }
        }
    }
    #[cfg(test)]
    assert_eq!(
        need,
        reference::future_need(index.unit(), owner),
        "sparse scheduler future need diverged from the dense fixed point"
    );
    need
}

/// Canonical verification deliberately owns a second fixed-point traversal;
/// it consumes no scheduler liveness or placement facts. It computes the same
/// least fixed point breadth first: the blocks that need the owner on their
/// own account, then every block that reaches one of them through blocks that
/// neither bind nor redefine the owner.
fn verifier_future_need(index: &UnitIndex<'_>, owner: OwnerId) -> BTreeSet<BlockId> {
    let mut result = BTreeSet::new();
    let mut frontier = std::collections::VecDeque::new();
    for (block, uses) in index.mentioning_blocks(owner) {
        if uses.param {
            continue;
        }
        let before_redefinition = uses.used_before_redefinition();
        let redefined = uses.redefines;
        let answer = before_redefinition
            || (!redefined
                && (uses.terminator_nonterminal || uses.terminator_fixed || uses.edge_use()));
        if answer {
            result.insert(block);
            frontier.push_back(block);
        }
    }
    while let Some(current) = frontier.pop_front() {
        for predecessor in index.incoming(current) {
            if result.contains(&predecessor.source) {
                continue;
            }
            let uses = index.mentions(owner, predecessor.source_position);
            if uses.param || uses.redefines {
                continue;
            }
            result.insert(predecessor.source);
            frontier.push_back(predecessor.source);
        }
    }
    #[cfg(test)]
    assert_eq!(
        result,
        reference::future_need(index.unit(), owner),
        "sparse canonical future need diverged from the dense fixed point"
    );
    result
}

/// Re-anchor every scheduled terminal in the host-site map: an in-block
/// terminal right after the site action of the operation it follows (or at
/// the front of its unit's function-entry site), an edge terminal right after
/// the control-edge action of its edge, or the `Jump` terminator action that
/// stands for it. Co-located terminals keep reverse definition order.
///
/// Each anchor is the first matching action in site and action order. Every
/// anchor is found in one index of the site map built before any insertion:
/// an inserted terminal action names a scheduled identity, which no anchor
/// names, so insertions never move which action anchors a terminal, only
/// where it sits. Each touched site is then rebuilt once with its insertions,
/// in the order the terminals were placed.
fn rebuild_host_sites(
    program: &OwnershipProgram,
    sites: &mut HostSiteMap,
    placed: &[(usize, Vec<Placed>)],
) -> Result<(), OwnershipError> {
    if sites.records.is_empty() {
        return Ok(());
    }
    #[cfg(test)]
    let expected = {
        let mut expected = sites.clone();
        reference::rebuild_host_sites(program, &mut expected, placed).map(|()| expected)
    };
    let result = rebuild_host_sites_indexed(program, sites, placed);
    #[cfg(test)]
    assert_eq!(
        result.as_ref().map(|()| &*sites),
        expected.as_ref(),
        "indexed host-site rebuild diverged from the sequential reference"
    );
    result
}

fn rebuild_host_sites_indexed(
    program: &OwnershipProgram,
    sites: &mut HostSiteMap,
    placed: &[(usize, Vec<Placed>)],
) -> Result<(), OwnershipError> {
    let ids = placed
        .iter()
        .flat_map(|(unit, terminals)| terminals.iter().map(move |terminal| (*unit, terminal.id)))
        .collect::<BTreeSet<_>>();
    for record in &mut sites.records {
        lookup_visited(record.actions.len());
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
    let anchors = SiteAnchors::new(sites);
    // Per site, the insertions in placement order: the slot is the position
    // in the site's actions, before any insertion, that the terminal goes in
    // front of.
    let mut insertions = BTreeMap::<usize, Vec<(usize, HostSiteAction)>>::new();
    for (unit_index, terminals) in placed {
        let unit = &program.units[*unit_index];
        // Each block, first by identity, with its non-scheduled operations.
        let mut blocks = BTreeMap::new();
        for block in &unit.blocks {
            lookup_visited(block.ops.len());
            blocks.entry(block.id).or_insert_with(|| {
                (
                    block,
                    non_scheduled_ops(block)
                        .map(|operation| operation.id)
                        .collect::<Vec<_>>(),
                )
            });
        }
        let mut terminals = terminals.iter().collect::<Vec<_>>();
        terminals
            .sort_by_key(|terminal| (terminal.point, std::cmp::Reverse(terminal.definition_order)));
        for terminal in terminals {
            lookup_visited(1);
            let (site_index, slot) = match terminal.point {
                Placement::OnEdge { source, edge } => {
                    let control = anchors.control_edge.get(&(*unit_index, edge, source));
                    let jump = blocks
                        .get(&source)
                        .is_some_and(|(block, _)| {
                            matches!(&block.terminator, Terminator::Jump(jump) if jump.id == edge)
                        })
                        .then(|| anchors.terminator.get(&(*unit_index, source)))
                        .flatten();
                    control
                        .into_iter()
                        .chain(jump)
                        .min()
                        .map(|&(index, position)| (index, position + 1))
                }
                Placement::InBlock { after: 0, .. } => anchors
                    .function_entry
                    .get(unit_index)
                    .map(|&index| (index, 0)),
                Placement::InBlock { block, after } => {
                    let (_, operations) =
                        blocks
                            .get(&block)
                            .ok_or_else(|| OwnershipError::MissingBlock {
                                unit: unit.name.clone(),
                                block: block.0,
                            })?;
                    let anchor =
                        *operations
                            .get(after - 1)
                            .ok_or_else(|| OwnershipError::HostSiteMap {
                                detail: format!(
                                    "scheduled terminal o{} has an invalid block anchor",
                                    terminal.id.0
                                ),
                            })?;
                    anchors
                        .operation
                        .get(&(*unit_index, block, anchor))
                        .map(|&(index, position)| (index, position + 1))
                }
            }
            .ok_or_else(|| OwnershipError::HostSiteMap {
                detail: format!(
                    "scheduled terminal o{} has no structural site",
                    terminal.id.0
                ),
            })?;
            let block = match terminal.point {
                Placement::InBlock { block, .. } | Placement::OnEdge { source: block, .. } => block,
            };
            insertions.entry(site_index).or_default().push((
                slot,
                HostSiteAction::Operation {
                    unit: *unit_index,
                    block,
                    operation: terminal.id,
                },
            ));
        }
    }
    for (site_index, mut inserted) in insertions {
        // Stable: terminals sharing a slot keep their placement order.
        inserted.sort_by_key(|(slot, _)| *slot);
        let record = &mut sites.records[site_index];
        let original = std::mem::take(&mut record.actions);
        lookup_visited(original.len() + inserted.len());
        let mut inserted = inserted.into_iter().peekable();
        let mut rebuilt = Vec::with_capacity(original.len() + inserted.len());
        for (position, action) in original.into_iter().enumerate() {
            while let Some((_, insertion)) = inserted.next_if(|(slot, _)| *slot == position) {
                rebuilt.push(insertion);
            }
            rebuilt.push(action);
        }
        rebuilt.extend(inserted.map(|(_, insertion)| insertion));
        record.actions = rebuilt;
    }
    Ok(())
}

/// The first site action, in site then action order, that each kind of
/// scheduled-terminal anchor can name.
struct SiteAnchors {
    /// `(unit, block, operation)` to the first `Operation` action.
    operation: BTreeMap<(usize, BlockId, OpId), (usize, usize)>,
    /// `(unit, edge, source)` to the first `ControlEdge` action.
    control_edge: BTreeMap<(usize, EdgeId, BlockId), (usize, usize)>,
    /// `(unit, block)` to the first `Terminator` action.
    terminator: BTreeMap<(usize, BlockId), (usize, usize)>,
    /// Each unit's first function-entry site.
    function_entry: BTreeMap<usize, usize>,
}

impl SiteAnchors {
    fn new(sites: &HostSiteMap) -> Self {
        let mut anchors = Self {
            operation: BTreeMap::new(),
            control_edge: BTreeMap::new(),
            terminator: BTreeMap::new(),
            function_entry: BTreeMap::new(),
        };
        for (index, record) in sites.records.iter().enumerate() {
            lookup_visited(record.actions.len());
            if record.kind == super::ir::HostSiteKind::FunctionEntry {
                anchors.function_entry.entry(record.unit).or_insert(index);
            }
            for (position, action) in record.actions.iter().enumerate() {
                let at = (index, position);
                match *action {
                    HostSiteAction::Operation {
                        unit,
                        block,
                        operation,
                    } => {
                        anchors
                            .operation
                            .entry((unit, block, operation))
                            .or_insert(at);
                    }
                    HostSiteAction::ControlEdge {
                        unit, edge, source, ..
                    } => {
                        anchors
                            .control_edge
                            .entry((unit, edge, source))
                            .or_insert(at);
                    }
                    HostSiteAction::Terminator { unit, block } => {
                        anchors.terminator.entry((unit, block)).or_insert(at);
                    }
                    HostSiteAction::Root { .. } => {}
                }
            }
        }
        anchors
    }
}

/// Every owner's scheduled terminals as they stand in the unit, in block,
/// operation and edge order: one pass for all owners.
fn actual_placements(unit: &Unit) -> BTreeMap<OwnerId, Vec<Placement>> {
    let mut result = BTreeMap::<OwnerId, Vec<Placement>>::new();
    for block in &unit.blocks {
        lookup_visited(block.ops.len());
        let mut non_scheduled = 0;
        for operation in &block.ops {
            if operation.role == OperationRole::ScheduledScopeExit {
                if let Ok(kind) = terminal_kind(unit, operation) {
                    result
                        .entry(kind.owner())
                        .or_default()
                        .push(Placement::InBlock {
                            block: block.id,
                            after: non_scheduled,
                        });
                }
            } else {
                non_scheduled += 1;
            }
        }
        for edge in block.terminator.edges() {
            lookup_visited(edge.terminals.len());
            for terminal in &edge.terminals {
                result
                    .entry(terminal.kind.owner())
                    .or_default()
                    .push(Placement::OnEdge {
                        source: block.id,
                        edge: edge.id,
                    });
            }
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
    index: &UnitIndex<'_>,
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
        .map(|placement| placement_point(index, unit, placement))
        .collect::<Result<BTreeSet<_>, _>>()?;
    frontier.extend(index.fixed_consume_points(owner));
    if frontier.len() == 1 {
        let definition = placement_point(
            index,
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

fn placement_point(
    index: &UnitIndex<'_>,
    unit: &Unit,
    placement: Placement,
) -> Result<SchedulePoint, OwnershipError> {
    match placement {
        Placement::OnEdge { edge, .. } => Ok(SchedulePoint::Edge(edge)),
        Placement::InBlock { block, after: 0 } => Ok(SchedulePoint::BlockEntry(block)),
        Placement::InBlock { block, after } => {
            let operation = index.anchor(block, after)?.ok_or_else(|| {
                invariant(unit, "terminal placement has no stable operation anchor")
            })?;
            Ok(SchedulePoint::AfterOperation { block, operation })
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
