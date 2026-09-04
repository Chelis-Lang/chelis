use std::collections::{BTreeMap, BTreeSet, VecDeque};

use chelis_types::manifest::{RootEntry, RootManifest};
use chelis_types::types::{Lane, Prim};

use crate::host::{
    ConcreteHostCallback, ConcreteHostCallbackKind, ConcreteHostExpr, ConcreteHostExprKind,
    ConcreteHostProgram, HostDisplayRoot, HostFunctionOrigin, HostTensorHelper,
};

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
    host: &ConcreteHostProgram,
    manifest: &RootManifest,
    program: &OwnershipProgram,
    sites: &HostSiteMap,
) -> Result<(), OwnershipError> {
    verify_host_payload_sites(host, manifest, sites)?;
    verify_host_actions(program, manifest, sites)
}

pub(super) fn verify_host_payload_sites(
    host: &ConcreteHostProgram,
    manifest: &RootManifest,
    sites: &HostSiteMap,
) -> Result<(), OwnershipError> {
    verify_materialized_roots(host, manifest)?;
    let expected = census_host_payload(host, manifest)?;
    if expected.len() != sites.records.len() {
        return Err(OwnershipError::HostSiteMap {
            detail: format!(
                "payload has {} structural sites, directive map has {}",
                expected.len(),
                sites.records.len()
            ),
        });
    }
    for (index, (expected_site, site)) in expected.iter().zip(&sites.records).enumerate() {
        if site.id.index() != index {
            return Err(OwnershipError::HostSiteMap {
                detail: format!("site at position {index} has a different opaque identity"),
            });
        }
        if site.unit != expected_site.unit {
            return Err(OwnershipError::HostSiteMap {
                detail: format!(
                    "payload site {index} belongs to unit {}, directive map routes it to unit {}",
                    expected_site.unit, site.unit
                ),
            });
        }
        if site.kind != expected_site.kind {
            return Err(OwnershipError::HostSiteMap {
                detail: format!(
                    "payload site {index} has kind {:?}, directive map has {:?}",
                    expected_site.kind, site.kind
                ),
            });
        }
    }
    Ok(())
}

pub(super) fn verify_host_actions(
    program: &OwnershipProgram,
    manifest: &RootManifest,
    sites: &HostSiteMap,
) -> Result<(), OwnershipError> {
    let mut operations = BTreeSet::new();
    let mut terminals = BTreeSet::new();
    let mut controls = BTreeMap::new();
    let mut roots = BTreeMap::new();
    let mut display_roots = Vec::new();
    for (index, site) in sites.records.iter().enumerate() {
        if matches!(
            site.kind,
            super::ir::HostSiteKind::Binding | super::ir::HostSiteKind::Argument
        ) && !site.actions.is_empty()
        {
            return Err(OwnershipError::HostSiteMap {
                detail: format!("structural site {index} carries an ownership action"),
            });
        }
        for action in &site.actions {
            let action_unit = match action {
                HostSiteAction::Operation { unit, .. }
                | HostSiteAction::Terminator { unit, .. }
                | HostSiteAction::ControlEdge { unit, .. }
                | HostSiteAction::Root { unit, .. } => *unit,
            };
            if action_unit != site.unit {
                return Err(site_error(
                    index,
                    "action unit differs from its enclosing structural site",
                ));
            }
            match *action {
                HostSiteAction::Operation {
                    unit,
                    block,
                    operation,
                } => {
                    if !matches!(
                        site.kind,
                        super::ir::HostSiteKind::Expression
                            | super::ir::HostSiteKind::FunctionEntry
                            | super::ir::HostSiteKind::FunctionReturn
                            | super::ir::HostSiteKind::ManifestRoot
                    ) {
                        return Err(site_error(index, "operation has inappropriate site kind"));
                    }
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
                    if !matches!(
                        site.kind,
                        super::ir::HostSiteKind::Expression
                            | super::ir::HostSiteKind::FunctionEntry
                            | super::ir::HostSiteKind::FunctionReturn
                    ) {
                        return Err(site_error(index, "terminator has inappropriate site kind"));
                    }
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
                HostSiteAction::Root {
                    unit,
                    manifest_index,
                    owner,
                } => {
                    if site.kind != super::ir::HostSiteKind::ManifestRoot {
                        return Err(site_error(index, "root has inappropriate site kind"));
                    }
                    if !program
                        .units
                        .get(unit)
                        .is_some_and(|unit| unit.owners.contains_key(&owner))
                    {
                        return Err(site_error(index, "directive names missing owner"));
                    }
                    if let Some(manifest_index) = manifest_index {
                        let Some(entry) = manifest.entries.get(manifest_index) else {
                            return Err(site_error(index, "root names missing manifest entry"));
                        };
                        if entry.lane != Lane::Host {
                            return Err(site_error(index, "root names a non-host manifest entry"));
                        }
                        if roots.insert(manifest_index, (index, owner)).is_some() {
                            return Err(site_error(index, "manifest root belongs to two sites"));
                        }
                    } else {
                        display_roots.push((index, owner));
                    }
                }
                HostSiteAction::ControlEdge {
                    unit,
                    source,
                    target,
                } => {
                    let expected_kind = expected_control_kind(program, unit, source, target)
                        .ok_or_else(|| {
                            site_error(index, "edge is not a typed control successor")
                        })?;
                    if site.kind != expected_kind {
                        return Err(site_error(
                            index,
                            "control edge has inappropriate site kind",
                        ));
                    }
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
                    if controls.insert((unit, source, target), site.kind).is_some() {
                        return Err(site_error(index, "control edge belongs to two sites"));
                    }
                }
            }
        }
        if matches!(
            site.kind,
            super::ir::HostSiteKind::BranchEdge
                | super::ir::HostSiteKind::MatchArm
                | super::ir::HostSiteKind::LoopEdge
        ) && site.actions.len() != 1
        {
            return Err(site_error(
                index,
                "control-edge site does not carry exactly one action",
            ));
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
    let expected_controls = collect_expected_controls(program)?;
    if controls != expected_controls {
        return Err(OwnershipError::HostSiteMap {
            detail: "ownership control-edge/site correspondence is not bijective".to_string(),
        });
    }
    let expected_roots = manifest
        .entries
        .iter()
        .enumerate()
        .filter_map(|(index, root)| (root.lane == Lane::Host).then_some(index))
        .collect::<Vec<_>>();
    if roots.keys().copied().collect::<Vec<_>>() != expected_roots {
        return Err(OwnershipError::HostSiteMap {
            detail: "manifest root/site correspondence is not bijective".to_string(),
        });
    }
    for (manifest_index, (site_index, owner)) in roots {
        let entry = &manifest.entries[manifest_index];
        let site = &sites.records[site_index];
        let consumes = site
            .actions
            .iter()
            .filter_map(|action| match *action {
                HostSiteAction::Operation {
                    unit,
                    block,
                    operation,
                } => program
                    .units
                    .get(unit)
                    .and_then(|unit| unit.blocks.iter().find(|candidate| candidate.id == block))
                    .and_then(|block| block.ops.get(operation)),
                _ => None,
            })
            .filter_map(|op| match op {
                Op::RootConsume { root, owner } => Some((root, owner)),
                _ => None,
            })
            .collect::<Vec<_>>();
        if consumes.len() != 1
            || consumes[0].0 != &entry.name
            || consumes[0].1.owner != owner
            || consumes[0].1.use_ != OwnershipUse::Move
        {
            return Err(site_error(
                site_index,
                "root action does not match its exact root-consume operation",
            ));
        }
    }
    for (site_index, owner) in display_roots {
        let site = &sites.records[site_index];
        let consumes = site
            .actions
            .iter()
            .filter_map(|action| match *action {
                HostSiteAction::Operation {
                    unit,
                    block,
                    operation,
                } => program
                    .units
                    .get(unit)
                    .and_then(|unit| unit.blocks.iter().find(|candidate| candidate.id == block))
                    .and_then(|block| block.ops.get(operation)),
                _ => None,
            })
            .filter_map(|op| match op {
                Op::RootConsume { owner, .. } => Some(owner),
                _ => None,
            })
            .collect::<Vec<_>>();
        if consumes.len() != 1
            || consumes[0].owner != owner
            || consumes[0].use_ != OwnershipUse::Move
        {
            return Err(site_error(
                site_index,
                "display root action does not match its exact root-consume operation",
            ));
        }
    }
    Ok(())
}

fn collect_expected_controls(
    program: &OwnershipProgram,
) -> Result<BTreeMap<(usize, BlockId, BlockId), super::ir::HostSiteKind>, OwnershipError> {
    let mut result = BTreeMap::new();
    for (unit_index, unit) in program.units.iter().enumerate() {
        for block in &unit.blocks {
            let mut insert = |target: BlockId, kind: super::ir::HostSiteKind| {
                if result
                    .insert((unit_index, block.id, target), kind)
                    .is_some()
                {
                    Err(OwnershipError::HostSiteMap {
                        detail: format!(
                            "unit {unit_index} block b{} repeats control target b{}",
                            block.id.0, target.0
                        ),
                    })
                } else {
                    Ok(())
                }
            };
            match &block.terminator {
                Terminator::Branch {
                    then_edge,
                    else_edge,
                    ..
                } => {
                    insert(then_edge.target, super::ir::HostSiteKind::BranchEdge)?;
                    insert(else_edge.target, super::ir::HostSiteKind::BranchEdge)?;
                }
                Terminator::Match { arms, .. } => {
                    for arm in arms {
                        insert(arm.target, super::ir::HostSiteKind::MatchArm)?;
                    }
                }
                Terminator::Loop {
                    body_edge,
                    exit_edge,
                    ..
                } => {
                    insert(body_edge.target, super::ir::HostSiteKind::LoopEdge)?;
                    insert(exit_edge.target, super::ir::HostSiteKind::LoopEdge)?;
                }
                Terminator::Jump(_) | Terminator::Return { .. } | Terminator::Exit => {}
            }
        }
    }
    Ok(result)
}

fn expected_control_kind(
    program: &OwnershipProgram,
    unit: usize,
    source: BlockId,
    target: BlockId,
) -> Option<super::ir::HostSiteKind> {
    collect_expected_controls(program)
        .ok()?
        .get(&(unit, source, target))
        .copied()
}

fn site_error(index: usize, detail: &'static str) -> OwnershipError {
    OwnershipError::HostSiteMap {
        detail: format!("site {index} {detail}"),
    }
}

/// Reconstruct the exact payload-site sequence without consulting the
/// ownership IR or the site builder. This is deliberately a second traversal
/// over the retained emission payload: verification would be circular if it
/// derived its expected sites from the lowering result it is checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExpectedHostSite {
    unit: usize,
    kind: super::ir::HostSiteKind,
}

fn expected_site(unit: usize, kind: super::ir::HostSiteKind) -> ExpectedHostSite {
    ExpectedHostSite { unit, kind }
}

fn census_host_payload(
    host: &ConcreteHostProgram,
    manifest: &RootManifest,
) -> Result<Vec<ExpectedHostSite>, OwnershipError> {
    let mut sites = Vec::new();
    for binding in &host.globals {
        sites.push(expected_site(0, super::ir::HostSiteKind::Binding));
        census_host_expr(&binding.value, &host.global_tensor_helpers, 0, &mut sites)?;
    }
    sites.extend(
        manifest
            .entries
            .iter()
            .filter(|entry| entry.lane == Lane::Host)
            .map(|_| expected_site(0, super::ir::HostSiteKind::ManifestRoot)),
    );
    sites.extend(
        host.globals
            .iter()
            .flat_map(|binding| {
                let unmatched = binding.display_roots.iter().filter(|display| {
                    !manifest.entries.iter().any(|entry| {
                        let selected = binding.name == entry.def_name
                            || matches!(
                                &binding.value.kind,
                                ConcreteHostExprKind::Call { function, args, .. }
                                    if function == &entry.def_name && args.is_empty()
                            );
                        entry.lane == Lane::Host
                            && selected
                            && independent_display_root(entry) == **display
                    })
                });
                let legacy = (binding.display_roots.is_empty() && binding.display_name.is_some())
                    .then_some(());
                unmatched.map(|_| ()).chain(legacy)
            })
            .map(|_| expected_site(0, super::ir::HostSiteKind::ManifestRoot)),
    );
    sites.push(expected_site(0, super::ir::HostSiteKind::FunctionReturn));

    for (function_index, function) in host.functions.iter().enumerate() {
        let unit = function_index + 1;
        sites.extend(
            function
                .params
                .iter()
                .map(|_| expected_site(unit, super::ir::HostSiteKind::Binding)),
        );
        sites.push(expected_site(unit, super::ir::HostSiteKind::FunctionEntry));
        if function.origin == HostFunctionOrigin::Authored {
            sites.extend(
                function
                    .params
                    .iter()
                    .map(|_| expected_site(unit, super::ir::HostSiteKind::Argument)),
            );
        }
        sites.push(expected_site(unit, super::ir::HostSiteKind::FunctionReturn));
        census_host_expr(&function.body, &function.tensor_helpers, unit, &mut sites)?;
    }
    Ok(sites)
}

fn census_host_expr(
    expr: &ConcreteHostExpr,
    helpers: &[HostTensorHelper],
    unit: usize,
    sites: &mut Vec<ExpectedHostSite>,
) -> Result<(), OwnershipError> {
    use super::ir::HostSiteKind;

    sites.push(expected_site(unit, HostSiteKind::Expression));
    match &expr.kind {
        ConcreteHostExprKind::Int(_)
        | ConcreteHostExprKind::Float(_)
        | ConcreteHostExprKind::Bool(_)
        | ConcreteHostExprKind::String(_)
        | ConcreteHostExprKind::Var(_, _)
        | ConcreteHostExprKind::Unit => {}
        ConcreteHostExprKind::List(items, _)
        | ConcreteHostExprKind::Tuple(items, _)
        | ConcreteHostExprKind::AdtConstruct { fields: items, .. }
        | ConcreteHostExprKind::Call { args: items, .. }
        | ConcreteHostExprKind::Builtin { args: items, .. } => {
            for item in items {
                sites.push(expected_site(unit, HostSiteKind::Argument));
                census_host_expr(item, helpers, unit, sites)?;
            }
        }
        ConcreteHostExprKind::AdtFieldAccess { base, .. } => {
            census_host_expr(base, helpers, unit, sites)?;
        }
        ConcreteHostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            census_host_expr(cond, helpers, unit, sites)?;
            sites.push(expected_site(unit, HostSiteKind::BranchEdge));
            sites.push(expected_site(unit, HostSiteKind::BranchEdge));
            census_host_expr(then_expr, helpers, unit, sites)?;
            census_host_expr(else_expr, helpers, unit, sites)?;
        }
        ConcreteHostExprKind::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            census_host_expr(scrutinee, helpers, unit, sites)?;
            sites.push(expected_site(unit, HostSiteKind::MatchArm));
            sites.push(expected_site(unit, HostSiteKind::MatchArm));
            sites.push(expected_site(unit, HostSiteKind::Binding));
            census_host_expr(some_expr, helpers, unit, sites)?;
            census_host_expr(none_expr, helpers, unit, sites)?;
        }
        ConcreteHostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            census_host_expr(scrutinee, helpers, unit, sites)?;
            sites.extend(
                (0..arms.len() + usize::from(default_expr.is_some()))
                    .map(|_| expected_site(unit, HostSiteKind::MatchArm)),
            );
            for arm in arms {
                sites.extend(
                    arm.bindings
                        .iter()
                        .map(|_| expected_site(unit, HostSiteKind::Binding)),
                );
                census_host_expr(&arm.expr, helpers, unit, sites)?;
            }
            if let Some(default_expr) = default_expr {
                census_host_expr(default_expr, helpers, unit, sites)?;
            }
        }
        ConcreteHostExprKind::Let { bindings, body, .. } => {
            for binding in bindings {
                sites.push(expected_site(unit, HostSiteKind::Binding));
                census_host_expr(&binding.value, helpers, unit, sites)?;
            }
            census_host_expr(body, helpers, unit, sites)?;
        }
        ConcreteHostExprKind::Map { callback, list, .. }
        | ConcreteHostExprKind::Filter { callback, list, .. }
        | ConcreteHostExprKind::Partition { callback, list, .. }
        | ConcreteHostExprKind::FlatMap { callback, list, .. } => {
            census_host_expr(list, helpers, unit, sites)?;
            sites.push(expected_site(unit, HostSiteKind::LoopEdge));
            sites.push(expected_site(unit, HostSiteKind::LoopEdge));
            sites.push(expected_site(unit, HostSiteKind::Binding));
            census_host_callback(callback, helpers, unit, sites)?;
        }
        ConcreteHostExprKind::Fold {
            callback,
            init,
            list,
            ..
        } => {
            census_host_expr(init, helpers, unit, sites)?;
            census_host_expr(list, helpers, unit, sites)?;
            sites.push(expected_site(unit, HostSiteKind::Binding));
            sites.push(expected_site(unit, HostSiteKind::LoopEdge));
            sites.push(expected_site(unit, HostSiteKind::LoopEdge));
            sites.push(expected_site(unit, HostSiteKind::Binding));
            census_host_callback(callback, helpers, unit, sites)?;
        }
        ConcreteHostExprKind::Scan {
            callback,
            init,
            list,
            ..
        } => {
            census_host_expr(init, helpers, unit, sites)?;
            census_host_expr(list, helpers, unit, sites)?;
            sites.push(expected_site(unit, HostSiteKind::Binding));
            sites.push(expected_site(unit, HostSiteKind::LoopEdge));
            sites.push(expected_site(unit, HostSiteKind::LoopEdge));
            sites.push(expected_site(unit, HostSiteKind::Binding));
            census_host_callback(callback, helpers, unit, sites)?;
        }
        ConcreteHostExprKind::WithSeed { seed, body, .. } => {
            sites.push(expected_site(unit, HostSiteKind::Argument));
            census_host_expr(seed, helpers, unit, sites)?;
            sites.push(expected_site(unit, HostSiteKind::Argument));
            census_host_expr(body, helpers, unit, sites)?;
        }
        ConcreteHostExprKind::TensorCall { helper, args, .. } => {
            let Some(helper) = helpers.get(*helper) else {
                return Err(OwnershipError::HostSiteMap {
                    detail: format!("payload names missing tensor helper {helper}"),
                });
            };
            if independent_identity_helper(helper) && args.len() == 1 {
                census_host_expr(&args[0], helpers, unit, sites)?;
            } else {
                for arg in args {
                    sites.push(expected_site(unit, HostSiteKind::Argument));
                    census_host_expr(arg, helpers, unit, sites)?;
                }
            }
        }
    }
    Ok(())
}

fn census_host_callback(
    callback: &ConcreteHostCallback,
    helpers: &[HostTensorHelper],
    unit: usize,
    sites: &mut Vec<ExpectedHostSite>,
) -> Result<(), OwnershipError> {
    match &callback.kind {
        ConcreteHostCallbackKind::Named { .. } => Ok(()),
        ConcreteHostCallbackKind::Inline { body, .. } => {
            census_host_expr(body, helpers, unit, sites)
        }
    }
}

fn independent_identity_helper(helper: &HostTensorHelper) -> bool {
    if helper.dag.roots().len() != 1 || helper.inputs.len() != 1 {
        return false;
    }
    let Some(node) = helper.dag.get(helper.dag.roots()[0]) else {
        return false;
    };
    matches!(
        &node.op,
        crate::dag::RiscOp::Load { name }
            if node.output_type == helper.output && helper.inputs[0].name == *name
    )
}

fn verify_materialized_roots(
    host: &ConcreteHostProgram,
    manifest: &RootManifest,
) -> Result<(), OwnershipError> {
    for binding in &host.globals {
        for display in &binding.display_roots {
            let candidates = manifest
                .entries
                .iter()
                .filter(|entry| {
                    let is_selected_binding = binding.name == entry.def_name
                        || matches!(
                            &binding.value.kind,
                            ConcreteHostExprKind::Call { function, args, .. }
                                if function == &entry.def_name && args.is_empty()
                        );
                    is_selected_binding && independent_display_root(entry) == *display
                })
                .count();
            if candidates != 1 {
                return Err(OwnershipError::HostSiteMap {
                    detail: format!(
                        "materialized payload root `{}` on binding `{}` has {candidates} exact manifest entries",
                        display.name, binding.name
                    ),
                });
            }
        }
    }

    let host_roots = manifest
        .entries
        .iter()
        .filter(|entry| entry.lane == Lane::Host)
        .collect::<Vec<_>>();
    let payload_root_count = host
        .globals
        .iter()
        .map(|binding| {
            binding
                .display_roots
                .iter()
                .filter(|display| {
                    host_roots.iter().any(|entry| {
                        let is_selected_binding = binding.name == entry.def_name
                            || matches!(
                                &binding.value.kind,
                                ConcreteHostExprKind::Call { function, args, .. }
                                    if function == &entry.def_name && args.is_empty()
                            );
                        is_selected_binding && independent_display_root(entry) == **display
                    })
                })
                .count()
        })
        .sum::<usize>();
    if payload_root_count != host_roots.len() {
        return Err(OwnershipError::HostSiteMap {
            detail: format!(
                "payload has {payload_root_count} materialized roots, manifest selects {}",
                host_roots.len()
            ),
        });
    }

    for entry in host_roots {
        let expected = independent_display_root(entry);
        let candidates = host
            .globals
            .iter()
            .filter(|binding| {
                let is_selected_binding = binding.name == entry.def_name
                    || matches!(
                        &binding.value.kind,
                        ConcreteHostExprKind::Call { function, args, .. }
                            if function == &entry.def_name && args.is_empty()
                    );
                is_selected_binding
                    && binding
                        .display_roots
                        .iter()
                        .filter(|root| **root == expected)
                        .count()
                        == 1
            })
            .count();
        if candidates != 1 {
            return Err(OwnershipError::HostSiteMap {
                detail: format!(
                    "manifest root `{}` has {candidates} exact materialized payload bindings",
                    entry.name
                ),
            });
        }
    }
    Ok(())
}

fn independent_display_root(root: &RootEntry) -> HostDisplayRoot {
    let short_def = root
        .def_name
        .rsplit_once("__")
        .map(|(_, tail)| tail)
        .or_else(|| root.def_name.rsplit_once('.').map(|(_, tail)| tail))
        .unwrap_or(root.def_name.as_str());
    let suffix = root.name.strip_prefix(root.def_name.as_str()).unwrap_or("");
    HostDisplayRoot {
        name: format!("{short_def}{suffix}"),
        path: root.path.clone(),
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
        Op::Define { dest, .. } | Op::Copy { dest, .. } | Op::LoopItem { dest, .. } => Some(*dest),
        Op::Apply { dest, .. } => *dest,
        Op::Project { .. } | Op::Drop { .. } | Op::RootConsume { .. } => None,
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
        Op::Project { source } => use_operand(
            unit,
            block.id,
            source,
            Some(OwnershipUse::Borrow),
            definitions,
            live,
        ),
        Op::LoopItem { dest, list } => {
            use_operand(
                unit,
                block.id,
                list,
                Some(OwnershipUse::Borrow),
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
        if param.mode != ParamMode::Owned
            && !passed.contains(&param.owner)
            && unit.owners[&param.owner].origin != OwnerOrigin::ExternalBorrow
        {
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
        if param.mode != ParamMode::Owned
            && (passed.contains(&param.owner)
                || unit.owners[&param.owner].origin != OwnerOrigin::ExternalBorrow)
        {
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
        OwnerOrigin::ExternalBorrow => None,
    };
    let valid = matches!(
        (source_root, unit.owners[&target.owner].origin),
        (Some(a), OwnerOrigin::BorrowedFrom(b)) if a == b
    ) || matches!(
        (source_root, unit.owners[&target.owner].origin),
        (None, OwnerOrigin::ExternalBorrow)
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
