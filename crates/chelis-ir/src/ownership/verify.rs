use std::collections::{BTreeMap, BTreeSet, VecDeque};

use chelis_types::types::Prim;

use super::classify::{classify, render_type};
use super::error::OwnershipError;
use super::ir::{
    Block, BlockId, Edge, HostSiteAction, HostSiteMap, Op, Operand, OwnerId, OwnerOrigin,
    OwnershipProgram, OwnershipUse, ParamMode, Terminator, Unit, UnitKind,
};

pub(super) fn verify(program: &OwnershipProgram) -> Result<(), OwnershipError> {
    let mut names = BTreeSet::new();
    let mut roots = 0;
    for unit in &program.units {
        if !names.insert(&unit.name) {
            return Err(OwnershipError::DuplicateIdentity {
                unit: unit.name.clone(),
                kind: "unit",
                id: 0,
            });
        }
        roots += usize::from(unit.kind == UnitKind::Roots);
        verify_unit(unit)?;
    }
    if roots == 1 {
        Ok(())
    } else {
        Err(OwnershipError::RootUnitCount { actual: roots })
    }
}

/// Prove that the exact payload-site universe and the ownership-directive
/// universe travel together. Every operation and terminal in the ownership IR
/// is owned by exactly one opaque host site; every structural payload site has
/// one directive-list entry.
pub(super) fn verify_host_sites(
    program: &OwnershipProgram,
    sites: &HostSiteMap,
) -> Result<(), OwnershipError> {
    let mut operations = BTreeSet::new();
    let mut terminals = BTreeSet::new();
    for (index, site) in sites.records.iter().enumerate() {
        if site.id.index() != index {
            return Err(OwnershipError::HostSiteMap {
                detail: format!("site at position {index} has a different opaque identity"),
            });
        }
        if site.actions.is_empty() {
            return Err(OwnershipError::HostSiteMap {
                detail: format!("site {index} has no directive entry"),
            });
        }
        for action in &site.actions {
            match *action {
                HostSiteAction::Structural => {}
                HostSiteAction::Operation {
                    unit,
                    block,
                    operation,
                } => {
                    let Some(unit_ref) = program.units.get(unit) else {
                        return Err(site_error(index, "operation names missing unit"));
                    };
                    let Some(block_ref) = unit_ref.blocks.iter().find(|item| item.id == block)
                    else {
                        return Err(site_error(index, "operation names missing block"));
                    };
                    if block_ref.ops.get(operation).is_none() {
                        return Err(site_error(index, "operation index is outside its block"));
                    }
                    if !operations.insert((unit, block, operation)) {
                        return Err(site_error(index, "operation belongs to two sites"));
                    }
                }
                HostSiteAction::Terminator { unit, block } => {
                    if !terminals.insert((unit, block)) {
                        return Err(site_error(index, "terminator belongs to two sites"));
                    }
                    let Some(unit_ref) = program.units.get(unit) else {
                        return Err(site_error(index, "terminator names missing unit"));
                    };
                    if !unit_ref.blocks.iter().any(|item| item.id == block) {
                        return Err(site_error(index, "terminator names missing block"));
                    }
                }
                HostSiteAction::Root(owner) => {
                    if !program
                        .units
                        .iter()
                        .any(|unit| unit.owners.contains_key(&owner))
                    {
                        return Err(site_error(index, "directive names missing owner"));
                    }
                }
                HostSiteAction::ControlEdge {
                    unit,
                    source,
                    target,
                } => {
                    let Some(unit_ref) = program.units.get(unit) else {
                        return Err(site_error(index, "edge names missing unit"));
                    };
                    let Some(block_ref) = unit_ref.blocks.iter().find(|item| item.id == source)
                    else {
                        return Err(site_error(index, "edge names missing source block"));
                    };
                    if !successors(&block_ref.terminator).contains(&target) {
                        return Err(site_error(index, "edge target is not a successor"));
                    }
                }
            }
        }
    }
    let expected_operations = program
        .units
        .iter()
        .enumerate()
        .flat_map(|(unit, value)| {
            value.blocks.iter().flat_map(move |block| {
                (0..block.ops.len()).map(move |operation| (unit, block.id, operation))
            })
        })
        .collect::<BTreeSet<_>>();
    if operations != expected_operations {
        return Err(OwnershipError::HostSiteMap {
            detail: "ownership operation/site correspondence is not bijective".to_string(),
        });
    }
    let expected_terminals = program
        .units
        .iter()
        .enumerate()
        .flat_map(|(unit, value)| value.blocks.iter().map(move |block| (unit, block.id)))
        .collect::<BTreeSet<_>>();
    if terminals != expected_terminals {
        return Err(OwnershipError::HostSiteMap {
            detail: "ownership terminator/site correspondence is not bijective".to_string(),
        });
    }
    Ok(())
}

fn site_error(index: usize, detail: &'static str) -> OwnershipError {
    OwnershipError::HostSiteMap {
        detail: format!("site {index} {detail}"),
    }
}

fn verify_unit(unit: &Unit) -> Result<(), OwnershipError> {
    let blocks = blocks(unit)?;
    if !blocks.contains_key(&unit.entry) {
        return Err(missing_block(unit, unit.entry));
    }
    let definitions = definitions(unit)?;
    for owner in unit.owners.keys() {
        if !definitions.contains(owner) {
            return Err(incomplete(unit, *owner, "definition"));
        }
    }
    for owner in &definitions {
        let info = unit
            .owners
            .get(owner)
            .ok_or_else(|| incomplete(unit, *owner, "metadata"))?;
        let derived =
            classify(&info.ty, info.placement).map_err(|error| OwnershipError::BadClass {
                unit: unit.name.clone(),
                owner: owner.0,
                detail: error.to_string(),
            })?;
        if derived != info.class {
            return Err(OwnershipError::BadClass {
                unit: unit.name.clone(),
                owner: owner.0,
                detail: format!(
                    "type {} derives {derived}, records {}",
                    render_type(&info.ty),
                    info.class
                ),
            });
        }
    }
    check_params(unit)?;
    check_reachable(unit, &blocks)?;

    let entry = blocks[&unit.entry];
    let initial: BTreeSet<_> = entry.params.iter().map(|p| p.owner).collect();
    let mut incoming = BTreeMap::from([(unit.entry, initial)]);
    let mut queue = VecDeque::from([unit.entry]);
    while let Some(id) = queue.pop_front() {
        let block = blocks[&id];
        let mut live = incoming[&id].clone();
        check_borrows(unit, block.id, &live)?;
        for op in &block.ops {
            verify_op(unit, block, op, &definitions, &mut live)?;
        }
        verify_terminator(
            unit,
            block,
            &definitions,
            &blocks,
            &live,
            &mut incoming,
            &mut queue,
        )?;
    }
    Ok(())
}

fn blocks(unit: &Unit) -> Result<BTreeMap<BlockId, &Block>, OwnershipError> {
    let mut result = BTreeMap::new();
    for block in &unit.blocks {
        if result.insert(block.id, block).is_some() {
            return Err(OwnershipError::DuplicateIdentity {
                unit: unit.name.clone(),
                kind: "block",
                id: block.id.0,
            });
        }
    }
    Ok(result)
}

fn definitions(unit: &Unit) -> Result<BTreeSet<OwnerId>, OwnershipError> {
    let mut result = BTreeSet::new();
    for block in &unit.blocks {
        for owner in block
            .params
            .iter()
            .map(|p| p.owner)
            .chain(block.ops.iter().filter_map(destination))
        {
            if !result.insert(owner) {
                return Err(OwnershipError::DuplicateIdentity {
                    unit: unit.name.clone(),
                    kind: "owner",
                    id: owner.0,
                });
            }
        }
    }
    Ok(result)
}

fn destination(op: &Op) -> Option<OwnerId> {
    match op {
        Op::Define { dest, .. } | Op::Copy { dest, .. } => Some(*dest),
        Op::Apply { dest, .. } => *dest,
        Op::Drop { .. } | Op::RootConsume { .. } => None,
    }
}

fn check_params(unit: &Unit) -> Result<(), OwnershipError> {
    for block in &unit.blocks {
        for param in &block.params {
            let origin = unit.owners[&param.owner].origin;
            let valid = matches!(
                (param.mode, origin),
                (ParamMode::Owned, OwnerOrigin::Owned)
                    | (ParamMode::Borrowed, OwnerOrigin::BorrowedFrom(_))
                    | (ParamMode::Borrowed, OwnerOrigin::InternalBorrow)
                    | (ParamMode::Borrowed, OwnerOrigin::ExternalBorrow)
                    | (ParamMode::EntryBorrow, OwnerOrigin::ExternalBorrow)
            ) && (param.mode != ParamMode::EntryBorrow || block.id == unit.entry);
            if !valid {
                return Err(OwnershipError::BadClass {
                    unit: unit.name.clone(),
                    owner: param.owner.0,
                    detail: format!("{:?} parameter has origin {origin:?}", param.mode),
                });
            }
        }
    }
    Ok(())
}

fn check_reachable(unit: &Unit, blocks: &BTreeMap<BlockId, &Block>) -> Result<(), OwnershipError> {
    let mut reached = BTreeSet::new();
    let mut queue = VecDeque::from([unit.entry]);
    while let Some(id) = queue.pop_front() {
        if !reached.insert(id) {
            continue;
        }
        let block = blocks.get(&id).ok_or_else(|| missing_block(unit, id))?;
        for target in successors(&block.terminator) {
            if !blocks.contains_key(&target) {
                return Err(missing_block(unit, target));
            }
            queue.push_back(target);
        }
    }
    if let Some(block) = unit.blocks.iter().find(|b| !reached.contains(&b.id)) {
        return Err(OwnershipError::UnreachableBlock {
            unit: unit.name.clone(),
            block: block.id.0,
        });
    }
    Ok(())
}

fn successors(terminator: &Terminator) -> Vec<BlockId> {
    match terminator {
        Terminator::Return { .. } | Terminator::Exit => Vec::new(),
        Terminator::Jump(edge) => vec![edge.target],
        Terminator::Branch {
            then_edge,
            else_edge,
            ..
        }
        | Terminator::Loop {
            body_edge: then_edge,
            exit_edge: else_edge,
            ..
        } => vec![then_edge.target, else_edge.target],
        Terminator::Match { arms, .. } => arms.iter().map(|edge| edge.target).collect(),
    }
}

fn verify_op(
    unit: &Unit,
    block: &Block,
    op: &Op,
    definitions: &BTreeSet<OwnerId>,
    live: &mut BTreeSet<OwnerId>,
) -> Result<(), OwnershipError> {
    match op {
        Op::Define { dest, .. } => define(unit, *dest, live),
        Op::Apply {
            dest,
            label,
            schema,
            args,
        } => {
            if args.len() != schema.operands.len() {
                return Err(OwnershipError::OperationArity {
                    unit: unit.name.clone(),
                    block: block.id.0,
                    label: label.clone(),
                    expected: schema.operands.len(),
                    actual: args.len(),
                });
            }
            check_mixed_uses(unit, block.id, args)?;
            for (arg, expected) in args.iter().zip(&schema.operands) {
                use_operand(unit, block.id, arg, Some(*expected), definitions, live)?;
            }
            match (*dest, schema.result) {
                (Some(dest), Some(expected)) => {
                    let actual = unit.owners[&dest].class;
                    if actual != expected {
                        return Err(OwnershipError::OperationResultClass {
                            unit: unit.name.clone(),
                            block: block.id.0,
                            label: label.clone(),
                            expected: expected.to_string(),
                            actual: actual.to_string(),
                        });
                    }
                    define(unit, dest, live)?;
                }
                (None, None) => {}
                (Some(dest), None) => {
                    return Err(OwnershipError::IncompleteOwner {
                        unit: unit.name.clone(),
                        owner: dest.0,
                        missing: "operation result schema",
                    });
                }
                (None, Some(_)) => {
                    return Err(OwnershipError::LoweringInvariant {
                        unit: unit.name.clone(),
                        detail: format!("operation `{label}` schema declares an absent result"),
                    });
                }
            }
            Ok(())
        }
        Op::Copy { dest, source } => {
            use_operand(
                unit,
                block.id,
                source,
                Some(OwnershipUse::Clone),
                definitions,
                live,
            )?;
            define(unit, *dest, live)
        }
        Op::Drop { owner } => {
            if !definitions.contains(&owner.owner) {
                return Err(incomplete(unit, owner.owner, "definition"));
            }
            if !unit.owners[&owner.owner].class.is_heap() {
                return Err(OwnershipError::NonHeapDrop {
                    unit: unit.name.clone(),
                    owner: owner.owner.0,
                    block: block.id.0,
                });
            }
            use_operand(
                unit,
                block.id,
                owner,
                Some(OwnershipUse::Move),
                definitions,
                live,
            )
        }
        Op::RootConsume { root, owner } => {
            if unit.kind != UnitKind::Roots {
                return Err(OwnershipError::RootSinkOutsideRoots {
                    unit: unit.name.clone(),
                    root: root.clone(),
                });
            }
            use_operand(
                unit,
                block.id,
                owner,
                Some(OwnershipUse::Move),
                definitions,
                live,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_terminator(
    unit: &Unit,
    block: &Block,
    definitions: &BTreeSet<OwnerId>,
    blocks: &BTreeMap<BlockId, &Block>,
    live: &BTreeSet<OwnerId>,
    incoming: &mut BTreeMap<BlockId, BTreeSet<OwnerId>>,
    queue: &mut VecDeque<BlockId>,
) -> Result<(), OwnershipError> {
    match &block.terminator {
        Terminator::Return { result } => {
            if unit.kind != UnitKind::Function {
                return Err(wrong_terminal(unit, "roots", "return"));
            }
            let mut next = live.clone();
            use_operand(
                unit,
                block.id,
                result,
                Some(OwnershipUse::Move),
                definitions,
                &mut next,
            )?;
            check_terminal(unit, block.id, &next)
        }
        Terminator::Exit => {
            if unit.kind != UnitKind::Roots {
                return Err(wrong_terminal(unit, "function", "exit"));
            }
            check_terminal(unit, block.id, live)
        }
        Terminator::Jump(edge) => transfer(
            unit,
            block,
            edge,
            definitions,
            blocks,
            live,
            incoming,
            queue,
        ),
        Terminator::Branch {
            condition,
            then_edge,
            else_edge,
        } => {
            let mut next = live.clone();
            use_operand(
                unit,
                block.id,
                condition,
                Some(OwnershipUse::Borrow),
                definitions,
                &mut next,
            )?;
            if !matches!(
                unit.owners[&condition.owner].ty,
                crate::host_type_state::ConcreteHostType::Scalar(Prim::Bool)
            ) {
                return Err(OwnershipError::NonBoolBranch {
                    unit: unit.name.clone(),
                    owner: condition.owner.0,
                    block: block.id.0,
                });
            }
            transfer(
                unit,
                block,
                then_edge,
                definitions,
                blocks,
                &next,
                incoming,
                queue,
            )?;
            transfer(
                unit,
                block,
                else_edge,
                definitions,
                blocks,
                &next,
                incoming,
                queue,
            )
        }
        Terminator::Match { scrutinee, arms } => {
            let mut next = live.clone();
            use_operand(
                unit,
                block.id,
                scrutinee,
                Some(OwnershipUse::Borrow),
                definitions,
                &mut next,
            )?;
            for edge in arms {
                transfer(
                    unit,
                    block,
                    edge,
                    definitions,
                    blocks,
                    &next,
                    incoming,
                    queue,
                )?;
            }
            Ok(())
        }
        Terminator::Loop {
            list,
            body_edge,
            exit_edge,
        } => {
            let mut next = live.clone();
            use_operand(
                unit,
                block.id,
                list,
                Some(OwnershipUse::Borrow),
                definitions,
                &mut next,
            )?;
            transfer(
                unit,
                block,
                body_edge,
                definitions,
                blocks,
                &next,
                incoming,
                queue,
            )?;
            transfer(
                unit,
                block,
                exit_edge,
                definitions,
                blocks,
                &next,
                incoming,
                queue,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn transfer(
    unit: &Unit,
    block: &Block,
    edge: &Edge,
    definitions: &BTreeSet<OwnerId>,
    blocks: &BTreeMap<BlockId, &Block>,
    live: &BTreeSet<OwnerId>,
    incoming: &mut BTreeMap<BlockId, BTreeSet<OwnerId>>,
    queue: &mut VecDeque<BlockId>,
) -> Result<(), OwnershipError> {
    let target = blocks
        .get(&edge.target)
        .ok_or_else(|| missing_block(unit, edge.target))?;
    if edge.args.len() != target.params.len() {
        return Err(OwnershipError::EdgeArity {
            unit: unit.name.clone(),
            block: block.id.0,
            target: edge.target.0,
            expected: target.params.len(),
            actual: edge.args.len(),
        });
    }
    check_mixed_uses(unit, block.id, &edge.args)?;
    let passed = edge.args.iter().map(|a| a.owner).collect::<BTreeSet<_>>();
    let mut next = live.clone();
    for param in &block.params {
        if param.mode != ParamMode::Owned && !passed.contains(&param.owner) {
            next.remove(&param.owner);
        }
    }
    for (arg, param) in edge.args.iter().zip(&target.params) {
        use_operand(
            unit,
            block.id,
            arg,
            Some(param.mode.use_()),
            definitions,
            &mut next,
        )?;
        check_borrow_edge(unit, arg.owner, param)?;
    }
    for param in &block.params {
        if param.mode != ParamMode::Owned {
            next.remove(&param.owner);
        }
    }
    next.extend(target.params.iter().map(|p| p.owner));
    check_borrows(unit, target.id, &next)?;
    if let Some(expected) = incoming.get(&target.id) {
        if expected != &next {
            return Err(OwnershipError::JoinMismatch {
                unit: unit.name.clone(),
                block: target.id.0,
                expected: ids(expected),
                actual: ids(&next),
            });
        }
    } else {
        incoming.insert(target.id, next);
        queue.push_back(target.id);
    }
    Ok(())
}

fn check_mixed_uses(
    unit: &Unit,
    block: BlockId,
    operands: &[Operand],
) -> Result<(), OwnershipError> {
    if let Some(moved) = operands.iter().find(|a| {
        a.use_ == OwnershipUse::Move
            && operands
                .iter()
                .any(|b| b.owner == a.owner && b.use_ == OwnershipUse::Borrow)
    }) {
        Err(OwnershipError::LiveBorrowAtConsume {
            unit: unit.name.clone(),
            owner: moved.owner.0,
            borrow: moved.owner.0,
            block: block.0,
        })
    } else {
        Ok(())
    }
}

fn check_borrow_edge(
    unit: &Unit,
    source: OwnerId,
    target: &super::ir::BlockParam,
) -> Result<(), OwnershipError> {
    if target.mode == ParamMode::Owned {
        return Ok(());
    }
    let source_root = match unit.owners[&source].origin {
        OwnerOrigin::Owned => Some(source),
        OwnerOrigin::BorrowedFrom(owner) => Some(owner),
        OwnerOrigin::InternalBorrow | OwnerOrigin::ExternalBorrow => None,
    };
    let valid = matches!(
        (source_root, unit.owners[&target.owner].origin),
        (Some(a), OwnerOrigin::BorrowedFrom(b)) if a == b
    ) || matches!(
        (source_root, unit.owners[&target.owner].origin),
        (
            None,
            OwnerOrigin::InternalBorrow | OwnerOrigin::ExternalBorrow
        )
    );
    if valid {
        Ok(())
    } else {
        Err(OwnershipError::BadClass {
            unit: unit.name.clone(),
            owner: target.owner.0,
            detail: format!("borrow edge from %{} has incompatible provenance", source.0),
        })
    }
}

fn use_operand(
    unit: &Unit,
    block: BlockId,
    operand: &Operand,
    expected: Option<OwnershipUse>,
    definitions: &BTreeSet<OwnerId>,
    live: &mut BTreeSet<OwnerId>,
) -> Result<(), OwnershipError> {
    if !definitions.contains(&operand.owner) {
        return Err(incomplete(unit, operand.owner, "definition"));
    }
    if !live.contains(&operand.owner) {
        return Err(OwnershipError::OwnerNotLive {
            unit: unit.name.clone(),
            owner: operand.owner.0,
            block: block.0,
        });
    }
    if operand.use_ == OwnershipUse::Clone && expected != Some(OwnershipUse::Clone) {
        return Err(wrong_use(unit, block, operand, "borrow or move"));
    }
    if let Some(expected) = expected
        && operand.use_ != expected
    {
        return Err(wrong_use(unit, block, operand, expected.name()));
    }
    if let OwnerOrigin::BorrowedFrom(source) = unit.owners[&operand.owner].origin
        && !live.contains(&source)
    {
        return Err(OwnershipError::OwnerNotLive {
            unit: unit.name.clone(),
            owner: source.0,
            block: block.0,
        });
    }
    if operand.use_ == OwnershipUse::Move {
        if unit.owners[&operand.owner].origin != OwnerOrigin::Owned {
            return Err(OwnershipError::BorrowConsumed {
                unit: unit.name.clone(),
                owner: operand.owner.0,
                block: block.0,
            });
        }
        if let Some(borrow) = live.iter().find(|candidate| {
            unit.owners[candidate].origin == OwnerOrigin::BorrowedFrom(operand.owner)
        }) {
            return Err(OwnershipError::LiveBorrowAtConsume {
                unit: unit.name.clone(),
                owner: operand.owner.0,
                borrow: borrow.0,
                block: block.0,
            });
        }
        live.remove(&operand.owner);
    }
    Ok(())
}

fn define(unit: &Unit, owner: OwnerId, live: &mut BTreeSet<OwnerId>) -> Result<(), OwnershipError> {
    if unit.owners[&owner].origin != OwnerOrigin::Owned {
        return Err(OwnershipError::BadClass {
            unit: unit.name.clone(),
            owner: owner.0,
            detail: "operation result is borrowed".to_string(),
        });
    }
    live.insert(owner);
    Ok(())
}

fn check_borrows(
    unit: &Unit,
    block: BlockId,
    live: &BTreeSet<OwnerId>,
) -> Result<(), OwnershipError> {
    for owner in live {
        if let OwnerOrigin::BorrowedFrom(source) = unit.owners[owner].origin
            && !live.contains(&source)
        {
            return Err(OwnershipError::OwnerNotLive {
                unit: unit.name.clone(),
                owner: source.0,
                block: block.0,
            });
        }
    }
    Ok(())
}

fn check_terminal(
    unit: &Unit,
    block: BlockId,
    live: &BTreeSet<OwnerId>,
) -> Result<(), OwnershipError> {
    if let Some(owner) = live
        .iter()
        .find(|owner| unit.owners[owner].origin == OwnerOrigin::Owned)
    {
        Err(OwnershipError::MissingTerminal {
            unit: unit.name.clone(),
            owner: owner.0,
            block: block.0,
        })
    } else {
        Ok(())
    }
}

fn ids(owners: &BTreeSet<OwnerId>) -> BTreeSet<u32> {
    owners.iter().map(|owner| owner.0).collect()
}

fn incomplete(unit: &Unit, owner: OwnerId, missing: &'static str) -> OwnershipError {
    OwnershipError::IncompleteOwner {
        unit: unit.name.clone(),
        owner: owner.0,
        missing,
    }
}

fn missing_block(unit: &Unit, block: BlockId) -> OwnershipError {
    OwnershipError::MissingBlock {
        unit: unit.name.clone(),
        block: block.0,
    }
}

fn wrong_use(
    unit: &Unit,
    block: BlockId,
    operand: &Operand,
    expected: &'static str,
) -> OwnershipError {
    OwnershipError::WrongUse {
        unit: unit.name.clone(),
        owner: operand.owner.0,
        block: block.0,
        expected,
        actual: operand.use_.name(),
    }
}

fn wrong_terminal(unit: &Unit, kind: &'static str, terminal: &'static str) -> OwnershipError {
    OwnershipError::WrongTerminal {
        unit: unit.name.clone(),
        kind,
        terminal,
    }
}
