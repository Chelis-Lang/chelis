use std::collections::{BTreeMap, BTreeSet, VecDeque};

use chelis_types::manifest::{RootEntry, RootManifest};
use chelis_types::types::{Lane, Prim};

use crate::host::{
    ConcreteHostCallback, ConcreteHostCallbackKind, ConcreteHostExpr, ConcreteHostExprKind,
    ConcreteHostProgram, HostDisplayRoot, HostFunctionOrigin, HostTensorHelper,
};
use crate::host_type_state::ConcreteHostType;

use super::classify::{Placement, ValueClass, classify, render_type};
use super::error::OwnershipError;
use super::ir::{
    ApplyKind, Block, BlockId, Edge, EdgeId, HostSiteAction, HostSiteMap, Op, OpId, Operand,
    OperationRole, OwnerId, OwnerOrigin, OwnershipProgram, OwnershipUse, ParamMode, Terminal,
    Terminator, Unit, UnitId, UnitKind,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct CallableSchema {
    operation: super::ir::OperationSchema,
    parameter_classes: Vec<ValueClass>,
}

#[derive(Debug, Clone, Copy)]
struct DirectCallLiveCost {
    callee: UnitId,
    carry: super::LiveByteBound,
}

#[derive(Debug)]
struct UnitLiveFacts {
    max_live_heap_owners: usize,
    local_max_live_bytes: super::LiveByteBound,
    direct_calls: Vec<DirectCallLiveCost>,
}

#[derive(Debug)]
pub(super) struct HostVerification {
    tail_calls: BTreeSet<(UnitId, OpId)>,
    live_set_bound: super::VerifiedLiveSetBound,
}

impl HostVerification {
    pub(super) fn is_tail_call(&self, unit: UnitId, operation: OpId) -> bool {
        self.tail_calls.contains(&(unit, operation))
    }

    pub(super) fn live_set_bound(&self) -> &super::VerifiedLiveSetBound {
        &self.live_set_bound
    }
}

pub(super) fn verify(program: &OwnershipProgram) -> Result<HostVerification, OwnershipError> {
    if super::last_use::has_schedule(program) {
        super::last_use::verify_canonical(program)?;
    }
    let mut names = BTreeSet::new();
    let mut units = BTreeMap::new();
    let mut roots = 0;
    for unit in &program.units {
        if units.insert(unit.id, unit).is_some() {
            return Err(OwnershipError::DuplicateIdentity {
                unit: unit.name.clone(),
                kind: "unit",
                id: unit.id.0,
            });
        }
        if !names.insert(&unit.name) {
            return Err(OwnershipError::DuplicateIdentity {
                unit: unit.name.clone(),
                kind: "unit",
                id: 0,
            });
        }
        roots += usize::from(unit.kind == UnitKind::Roots);
    }
    let mut callable_schemas = BTreeMap::new();
    for unit in &program.units {
        if let Some(schema) = derive_callable_schema(unit)? {
            callable_schemas.insert(unit.id, schema);
        }
    }
    let mut live_facts = BTreeMap::new();
    let mut max_live_heap_owners = 0;
    for unit in &program.units {
        let facts = verify_unit(unit, &units, &callable_schemas)?;
        max_live_heap_owners = max_live_heap_owners.max(facts.max_live_heap_owners);
        live_facts.insert(unit.id, facts);
    }
    if roots != 1 {
        return Err(OwnershipError::RootUnitCount { actual: roots });
    }
    let tail_calls = program.units.iter().flat_map(classify_tail_calls).collect();
    Ok(HostVerification {
        tail_calls,
        live_set_bound: super::VerifiedLiveSetBound {
            max_live_heap_owners,
            max_live_bytes: compose_live_byte_bound(&program.units, &live_facts)?,
        },
    })
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
                    let ordinary_site = matches!(
                        site.kind,
                        super::ir::HostSiteKind::Expression
                            | super::ir::HostSiteKind::FunctionEntry
                            | super::ir::HostSiteKind::FunctionReturn
                            | super::ir::HostSiteKind::ManifestRoot
                    );
                    let Some(unit_ref) = program.units.get(unit) else {
                        return Err(site_error(index, "operation names missing unit"));
                    };
                    if !unit_ref.blocks.iter().any(|item| item.id == block) {
                        return Err(site_error(index, "operation names missing block"));
                    }
                    if !operation_exists_in_block(unit_ref, block, operation) {
                        return Err(site_error(index, "operation identity is outside its block"));
                    }
                    let is_edge_terminal = edge_for_operation(unit_ref, block, operation).is_some();
                    if (is_edge_terminal
                        && !operation_is_on_site_edge(program, site, unit, block, operation))
                        || (!is_edge_terminal && !ordinary_site)
                    {
                        return Err(site_error(index, "operation has inappropriate site kind"));
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
                    edge,
                    source,
                    target,
                } => {
                    let expected_kind = expected_control_kind(program, unit, edge, source, target)
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
                    if !block_ref
                        .terminator
                        .edges()
                        .any(|candidate| candidate.id == edge && candidate.target == target)
                    {
                        return Err(site_error(index, "edge target is not a successor"));
                    }
                    if controls.insert((unit, edge), site.kind).is_some() {
                        return Err(site_error(index, "control edge belongs to two sites"));
                    }
                }
            }
        }
        if is_control_site(site.kind)
            && (!matches!(
                site.actions.first(),
                Some(HostSiteAction::ControlEdge { .. })
            ) || site
                .actions
                .iter()
                .skip(1)
                .any(|action| !matches!(action, HostSiteAction::Operation { .. })))
        {
            return Err(site_error(
                index,
                "control-edge site must carry its edge before edge-local terminals",
            ));
        }
    }
    let expected_operations = program
        .units
        .iter()
        .enumerate()
        .flat_map(|(unit, value)| {
            value.blocks.iter().flat_map(move |block| {
                block
                    .ops
                    .iter()
                    .map(move |operation| (unit, block.id, operation.id))
                    .chain(
                        block
                            .terminator
                            .edges()
                            .flat_map(|edge| &edge.terminals)
                            .map(move |terminal| (unit, block.id, terminal.id)),
                    )
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
                    .and_then(|block| block.ops.iter().find(|candidate| candidate.id == operation))
                    .map(|operation| &operation.kind),
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
                    .and_then(|block| block.ops.iter().find(|candidate| candidate.id == operation))
                    .map(|operation| &operation.kind),
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

fn is_control_site(kind: super::ir::HostSiteKind) -> bool {
    matches!(
        kind,
        super::ir::HostSiteKind::BranchEdge
            | super::ir::HostSiteKind::MatchArm
            | super::ir::HostSiteKind::LoopEdge
    )
}

fn operation_is_on_site_edge(
    program: &OwnershipProgram,
    site: &super::ir::HostSiteRecord,
    unit: usize,
    block: BlockId,
    operation: OpId,
) -> bool {
    let Some(unit_ref) = program.units.get(unit) else {
        return false;
    };
    let Some(edge) = edge_for_operation(unit_ref, block, operation) else {
        return false;
    };
    let Some(operation_index) = site.actions.iter().position(
        |action| matches!(action, HostSiteAction::Operation { operation: id, .. } if *id == operation),
    ) else {
        return false;
    };
    site.actions[..operation_index]
        .iter()
        .any(|action| match *action {
            HostSiteAction::ControlEdge {
                unit: action_unit,
                edge: action_edge,
                source,
                target,
            } => {
                action_unit == unit
                    && source == block
                    && action_edge == edge.id
                    && target == edge.target
            }
            HostSiteAction::Terminator {
                unit: action_unit,
                block: action_block,
            } => {
                action_unit == unit
                    && action_block == block
                    && matches!(
                        unit_ref
                            .blocks
                            .iter()
                            .find(|candidate| candidate.id == block)
                            .map(|block| &block.terminator),
                        Some(Terminator::Jump(jump)) if jump.id == edge.id
                    )
            }
            _ => false,
        })
}

fn edge_for_operation(unit: &Unit, block: BlockId, operation: OpId) -> Option<&Edge> {
    unit.blocks
        .iter()
        .find(|candidate| candidate.id == block)?
        .terminator
        .edges()
        .find(|edge| {
            edge.terminals
                .iter()
                .any(|terminal| terminal.id == operation)
        })
}

fn collect_expected_controls(
    program: &OwnershipProgram,
) -> Result<BTreeMap<(usize, EdgeId), super::ir::HostSiteKind>, OwnershipError> {
    let mut result = BTreeMap::new();
    for (unit_index, unit) in program.units.iter().enumerate() {
        for block in &unit.blocks {
            let mut insert = |edge: &Edge, kind: super::ir::HostSiteKind| {
                if result.insert((unit_index, edge.id), kind).is_some() {
                    Err(OwnershipError::HostSiteMap {
                        detail: format!(
                            "unit {unit_index} repeats control edge e{} from b{} to b{}",
                            edge.id.0, block.id.0, edge.target.0
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
                    insert(then_edge, super::ir::HostSiteKind::BranchEdge)?;
                    insert(else_edge, super::ir::HostSiteKind::BranchEdge)?;
                }
                Terminator::Match { arms, .. } => {
                    for arm in arms {
                        insert(arm, super::ir::HostSiteKind::MatchArm)?;
                    }
                }
                Terminator::Loop {
                    body_edge,
                    exit_edge,
                    ..
                } => {
                    insert(body_edge, super::ir::HostSiteKind::LoopEdge)?;
                    insert(exit_edge, super::ir::HostSiteKind::LoopEdge)?;
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
    edge: EdgeId,
    source: BlockId,
    target: BlockId,
) -> Option<super::ir::HostSiteKind> {
    let unit_ref = program.units.get(unit)?;
    let block = unit_ref.blocks.iter().find(|block| block.id == source)?;
    block
        .terminator
        .edges()
        .find(|candidate| candidate.id == edge && candidate.target == target)?;
    collect_expected_controls(program)
        .ok()?
        .get(&(unit, edge))
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
#[derive(Debug, Clone, Copy)]
struct ExpectedHostSite<'a> {
    unit: usize,
    kind: super::ir::HostSiteKind,
    #[cfg(feature = "lowering-trace")]
    expression: Option<&'a ConcreteHostExpr>,
    #[cfg(not(feature = "lowering-trace"))]
    source_lifetime: std::marker::PhantomData<&'a ()>,
}

fn expected_site(unit: usize, kind: super::ir::HostSiteKind) -> ExpectedHostSite<'static> {
    ExpectedHostSite {
        unit,
        kind,
        #[cfg(feature = "lowering-trace")]
        expression: None,
        #[cfg(not(feature = "lowering-trace"))]
        source_lifetime: std::marker::PhantomData,
    }
}

fn census_host_payload<'a>(
    host: &'a ConcreteHostProgram,
    manifest: &RootManifest,
) -> Result<Vec<ExpectedHostSite<'a>>, OwnershipError> {
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

fn census_host_expr<'a>(
    expr: &'a ConcreteHostExpr,
    helpers: &'a [HostTensorHelper],
    unit: usize,
    sites: &mut Vec<ExpectedHostSite<'a>>,
) -> Result<(), OwnershipError> {
    use super::ir::HostSiteKind;

    let expression_site = expected_site(unit, HostSiteKind::Expression);
    #[cfg(feature = "lowering-trace")]
    let expression_site = ExpectedHostSite {
        expression: Some(expr),
        ..expression_site
    };
    sites.push(expression_site);
    match &expr.kind {
        ConcreteHostExprKind::ResultClaimScope { body, .. } => {
            census_host_expr(body, helpers, unit, sites)?;
        }
        ConcreteHostExprKind::FormalIngress { value, .. }
        | ConcreteHostExprKind::ExtentSites { value, .. } => {
            census_host_expr(value, helpers, unit, sites)?;
        }
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
        ConcreteHostExprKind::SignatureEntry { args, lists, .. } => {
            for item in args.iter().chain(lists.iter().map(|entry| &entry.value)) {
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
        ConcreteHostExprKind::Let { bindings, body, .. }
        | ConcreteHostExprKind::RetainedInvocation { bindings, body, .. } => {
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
        ConcreteHostExprKind::TensorCall { helper, args, .. } => {
            let Some(helper) = helpers.get(*helper) else {
                return Err(OwnershipError::HostSiteMap {
                    detail: format!("payload names missing tensor helper {helper}"),
                });
            };
            if helper.identity_input().is_some() && args.len() == 1 {
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

/// Project only the already-verified, retained source-structural census. No
/// emitted spelling or separately supplied source may construct this cursor.
#[cfg(feature = "lowering-trace")]
pub(super) fn source_expressions(
    emission: super::VerifiedHostEmission<'_>,
) -> Vec<super::VerifiedHostSourceSite<'_>> {
    let expected = census_host_payload(&emission.payload.program, &emission.payload.manifest)
        .expect("verified host payload census");
    expected
        .into_iter()
        .zip(emission.sites())
        .filter_map(|(expected, site)| {
            expected
                .expression
                .map(|expression| super::VerifiedHostSourceSite { site, expression })
        })
        .collect()
}

fn census_host_callback<'a>(
    callback: &'a ConcreteHostCallback,
    helpers: &'a [HostTensorHelper],
    unit: usize,
    sites: &mut Vec<ExpectedHostSite<'a>>,
) -> Result<(), OwnershipError> {
    match &callback.kind {
        ConcreteHostCallbackKind::Named { .. } => Ok(()),
        ConcreteHostCallbackKind::Inline { body, .. } => {
            census_host_expr(body, helpers, unit, sites)
        }
    }
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
    let short_def = if chelis_types::is_linker_format_name(&root.def_name) {
        chelis_types::demangle_ident(&root.def_name)
    } else {
        root.def_name.clone()
    };
    let suffix = root.name.strip_prefix(root.def_name.as_str()).unwrap_or("");
    HostDisplayRoot {
        name: format!("{short_def}{suffix}"),
        path: root.path.clone(),
    }
}

fn verify_unit(
    unit: &Unit,
    units: &BTreeMap<UnitId, &Unit>,
    callable_schemas: &BTreeMap<UnitId, CallableSchema>,
) -> Result<UnitLiveFacts, OwnershipError> {
    if !matches!(
        (unit.kind, unit.callable_body),
        (UnitKind::Roots, None) | (UnitKind::Function, Some(_))
    ) {
        return Err(OwnershipError::LoweringInvariant {
            unit: unit.name.clone(),
            detail: "ownership unit kind and callable body disagree".to_string(),
        });
    }
    let blocks = blocks(unit)?;
    if !blocks.contains_key(&unit.entry) {
        return Err(missing_block(unit, unit.entry));
    }
    let definitions = definitions(unit)?;
    check_operation_ids(unit)?;
    for block in &unit.blocks {
        for operation in &block.ops {
            if operation.role != OperationRole::Semantic
                && !matches!(operation.kind, Op::Drop { .. } | Op::Discard { .. })
            {
                return Err(OwnershipError::LoweringInvariant {
                    unit: unit.name.clone(),
                    detail: format!(
                        "operation o{} marks a non-terminal as a scope exit",
                        operation.id.0
                    ),
                });
            }
        }
    }
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
    let mut max_live_heap_owners = 0;
    let mut local_max_live_bytes = super::LiveByteBound::Exact(0);
    let mut direct_calls = Vec::new();
    while let Some(id) = queue.pop_front() {
        let block = blocks[&id];
        let mut live = incoming[&id].clone();
        check_borrows(unit, block.id, &live)?;
        max_live_heap_owners = max_live_heap_owners.max(live_heap_count(unit, &live));
        local_max_live_bytes = local_max_live_bytes.maximum(live_byte_cost(unit, &live)?);
        for operation in &block.ops {
            if let Op::Apply {
                kind: ApplyKind::DirectCall { callee },
                args,
                ..
            } = &operation.kind
            {
                let mut carry = live.clone();
                for arg in args {
                    if arg.use_ == OwnershipUse::Move {
                        carry.remove(&arg.owner);
                    }
                }
                direct_calls.push(DirectCallLiveCost {
                    callee: *callee,
                    carry: live_byte_cost(unit, &carry)?,
                });
            }
            if let Some(dest) = destination(&operation.kind)
                && unit.owners[&dest].class.is_heap()
            {
                // Account for the transient point after result allocation and
                // before moved inputs receive their post-operation terminal.
                max_live_heap_owners = max_live_heap_owners.max(live_heap_count(unit, &live) + 1);
                if !matches!(
                    &operation.kind,
                    Op::Apply {
                        kind: ApplyKind::DirectCall { .. },
                        ..
                    }
                ) {
                    local_max_live_bytes = local_max_live_bytes.maximum(add_live_byte_bounds(
                        live_byte_cost(unit, &live)?,
                        owner_byte_cost(unit, dest)?,
                        || {
                            format!(
                                "accounting for `{}` operation o{} result",
                                unit.name, operation.id.0
                            )
                        },
                    )?);
                }
            }
            verify_op(
                unit,
                block,
                &operation.kind,
                &definitions,
                units,
                callable_schemas,
                &mut live,
            )?;
            max_live_heap_owners = max_live_heap_owners.max(live_heap_count(unit, &live));
            local_max_live_bytes = local_max_live_bytes.maximum(live_byte_cost(unit, &live)?);
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
    Ok(UnitLiveFacts {
        max_live_heap_owners,
        local_max_live_bytes,
        direct_calls,
    })
}

fn derive_callable_schema(unit: &Unit) -> Result<Option<CallableSchema>, OwnershipError> {
    let body = match (unit.kind, unit.callable_body) {
        (UnitKind::Roots, None) => return Ok(None),
        (UnitKind::Function, Some(body)) => body,
        _ => {
            return Err(OwnershipError::LoweringInvariant {
                unit: unit.name.clone(),
                detail: "ownership unit kind and callable body disagree".to_string(),
            });
        }
    };
    let blocks = blocks(unit)?;
    let reached = reachable_blocks(unit, &blocks)?;
    if !reached.contains(&body.entry()) {
        return Err(OwnershipError::UnreachableBlock {
            unit: unit.name.clone(),
            block: body.entry().0,
        });
    }
    let callable = blocks
        .get(&body.entry())
        .ok_or_else(|| missing_block(unit, body.entry()))?;
    let mut operands = Vec::with_capacity(callable.params.len());
    let mut parameter_classes = Vec::with_capacity(callable.params.len());
    let mut saw_capture = false;
    for param in &callable.params {
        if param.mode == ParamMode::EntryBorrow {
            return Err(OwnershipError::LoweringInvariant {
                unit: unit.name.clone(),
                detail: format!(
                    "callable body b{} carries artifact-only EntryBorrow parameter %{}",
                    callable.id.0, param.owner.0
                ),
            });
        }
        let info = unit
            .owners
            .get(&param.owner)
            .ok_or_else(|| incomplete(unit, param.owner, "metadata"))?;
        match info.placement {
            Placement::Parameter if saw_capture => {
                return Err(OwnershipError::LoweringInvariant {
                    unit: unit.name.clone(),
                    detail: format!(
                        "callable parameter %{} follows captured environment parameters",
                        param.owner.0
                    ),
                });
            }
            Placement::Parameter => {
                operands.push(param.mode.use_());
                parameter_classes.push(info.class);
            }
            Placement::Value => {
                saw_capture = true;
                if param.mode != ParamMode::Borrowed || info.origin != OwnerOrigin::ExternalBorrow {
                    return Err(OwnershipError::LoweringInvariant {
                        unit: unit.name.clone(),
                        detail: format!(
                            "callable environment parameter %{} is not an external borrow",
                            param.owner.0
                        ),
                    });
                }
            }
        }
    }

    let mut result: Option<ValueClass> = None;
    for block in unit
        .blocks
        .iter()
        .filter(|block| reached.contains(&block.id))
    {
        let Terminator::Return { result: returned } = &block.terminator else {
            continue;
        };
        if returned.use_ != OwnershipUse::Move {
            return Err(OwnershipError::FunctionReturnMode {
                unit: unit.name.clone(),
                block: block.id.0,
                actual: returned.use_.name(),
            });
        }
        let actual = unit
            .owners
            .get(&returned.owner)
            .ok_or_else(|| incomplete(unit, returned.owner, "metadata"))?
            .class;
        if let Some(expected) = result
            && expected != actual
        {
            return Err(OwnershipError::FunctionReturnClass {
                unit: unit.name.clone(),
                block: block.id.0,
                expected: expected.to_string(),
                actual: actual.to_string(),
            });
        }
        result = Some(actual);
    }
    let result = result.ok_or_else(|| OwnershipError::LoweringInvariant {
        unit: unit.name.clone(),
        detail: "function has no reachable return".to_string(),
    })?;
    Ok(Some(CallableSchema {
        operation: super::ir::OperationSchema::new(operands, Some(result)),
        parameter_classes,
    }))
}

fn check_operation_ids(unit: &Unit) -> Result<(), OwnershipError> {
    let mut operations = BTreeSet::new();
    let mut edges = BTreeSet::new();
    for block in &unit.blocks {
        for operation in &block.ops {
            if !operations.insert(operation.id) {
                return Err(OwnershipError::DuplicateIdentity {
                    unit: unit.name.clone(),
                    kind: "operation",
                    id: operation.id.0,
                });
            }
        }
        for edge in block.terminator.edges() {
            if edge.id == EdgeId::UNASSIGNED || !edges.insert(edge.id) {
                return Err(OwnershipError::DuplicateIdentity {
                    unit: unit.name.clone(),
                    kind: "edge",
                    id: edge.id.0,
                });
            }
            for terminal in &edge.terminals {
                if !operations.insert(terminal.id) {
                    return Err(OwnershipError::DuplicateIdentity {
                        unit: unit.name.clone(),
                        kind: "operation",
                        id: terminal.id.0,
                    });
                }
            }
        }
    }
    Ok(())
}

fn operation_exists_in_block(unit: &Unit, block: BlockId, operation: OpId) -> bool {
    unit.blocks
        .iter()
        .find(|candidate| candidate.id == block)
        .is_some_and(|block| {
            block.ops.iter().any(|candidate| candidate.id == operation)
                || block
                    .terminator
                    .edges()
                    .flat_map(|edge| &edge.terminals)
                    .any(|terminal| terminal.id == operation)
        })
}

fn live_heap_count(unit: &Unit, live: &BTreeSet<OwnerId>) -> usize {
    live.iter()
        .filter(|owner| {
            unit.owners[owner].origin == OwnerOrigin::Owned && unit.owners[owner].class.is_heap()
        })
        .count()
}

/// Sum two live-byte bounds, reporting `context` only when the addition
/// overflows.
///
/// `context` is a closure rather than a `String` because the call inside
/// [`live_byte_cost`] runs once per live owner per operation, and
/// `live_byte_cost` itself runs two to four times per operation: an eagerly
/// formatted diagnostic put roughly 39% of that function's samples in
/// `alloc::fmt::format::format_inner` at N=640 on the chelis#1205 corpus, for
/// a string the success path discards. The diagnostic text is unchanged
/// (chelis#2331).
fn add_live_byte_bounds(
    lhs: super::LiveByteBound,
    rhs: super::LiveByteBound,
    context: impl FnOnce() -> String,
) -> Result<super::LiveByteBound, OwnershipError> {
    match (lhs, rhs) {
        (super::LiveByteBound::Unbounded, _) | (_, super::LiveByteBound::Unbounded) => {
            Ok(super::LiveByteBound::Unbounded)
        }
        (super::LiveByteBound::Unknown, _) | (_, super::LiveByteBound::Unknown) => {
            Ok(super::LiveByteBound::Unknown)
        }
        (super::LiveByteBound::Exact(lhs), super::LiveByteBound::Exact(rhs)) => lhs
            .checked_add(rhs)
            .map(super::LiveByteBound::Exact)
            .ok_or_else(|| OwnershipError::LiveByteBoundOverflow { context: context() }),
    }
}

fn owner_byte_cost(unit: &Unit, owner: OwnerId) -> Result<super::LiveByteBound, OwnershipError> {
    let info = unit
        .owners
        .get(&owner)
        .ok_or_else(|| incomplete(unit, owner, "metadata"))?;
    let ConcreteHostType::Tensor(tensor) = &info.ty else {
        return Ok(if info.class.is_heap() {
            super::LiveByteBound::Unknown
        } else {
            super::LiveByteBound::Exact(0)
        });
    };
    if tensor.precision == Prim::String {
        return Ok(super::LiveByteBound::Unknown);
    }
    let dtype =
        tensor
            .precision
            .runtime_dtype()
            .map_err(|_| OwnershipError::LiveByteBoundDType {
                unit: unit.name.clone(),
                owner: owner.0,
                dtype: tensor.precision.name(),
            })?;
    let width =
        u64::try_from(dtype.byte_width()).map_err(|_| OwnershipError::LiveByteBoundOverflow {
            context: format!(
                "converting `{}` owner %{} runtime dtype width",
                unit.name, owner.0
            ),
        })?;
    let mut elements = 1u64;
    for dim in &tensor.dims {
        let value = match dim {
            crate::dag::DimInfo::Lit(value) | crate::dag::DimInfo::Named(_, Some(value)) => {
                u64::try_from(*value).map_err(|_| OwnershipError::LiveByteBoundOverflow {
                    context: format!(
                        "converting `{}` owner %{} tensor dimension {value}",
                        unit.name, owner.0
                    ),
                })?
            }
            crate::dag::DimInfo::Named(_, None) => return Ok(super::LiveByteBound::Unknown),
        };
        elements =
            elements
                .checked_mul(value)
                .ok_or_else(|| OwnershipError::LiveByteBoundOverflow {
                    context: format!(
                        "multiplying `{}` owner %{} tensor dimensions",
                        unit.name, owner.0
                    ),
                })?;
    }
    elements
        .checked_mul(width)
        .map(super::LiveByteBound::Exact)
        .ok_or_else(|| OwnershipError::LiveByteBoundOverflow {
            context: format!(
                "multiplying `{}` owner %{} tensor elements by dtype width",
                unit.name, owner.0
            ),
        })
}

fn live_byte_cost(
    unit: &Unit,
    live: &BTreeSet<OwnerId>,
) -> Result<super::LiveByteBound, OwnershipError> {
    let mut result = super::LiveByteBound::Exact(0);
    for owner in live {
        let info = unit
            .owners
            .get(owner)
            .ok_or_else(|| incomplete(unit, *owner, "metadata"))?;
        if info.origin != OwnerOrigin::Owned || !info.class.is_heap() {
            continue;
        }
        result = add_live_byte_bounds(result, owner_byte_cost(unit, *owner)?, || {
            format!("summing live owners in `{}`", unit.name)
        })?;
    }
    Ok(result)
}

fn unit_reaches(start: UnitId, target: UnitId, facts: &BTreeMap<UnitId, UnitLiveFacts>) -> bool {
    let mut pending = vec![start];
    let mut visited = BTreeSet::new();
    while let Some(unit) = pending.pop() {
        if unit == target {
            return true;
        }
        if visited.insert(unit)
            && let Some(next) = facts.get(&unit)
        {
            pending.extend(next.direct_calls.iter().map(|call| call.callee));
        }
    }
    false
}

#[derive(Debug)]
struct ComponentLiveFacts {
    local: super::LiveByteBound,
    outgoing: Vec<(usize, super::LiveByteBound)>,
}

fn compose_live_byte_bound(
    units: &[Unit],
    facts: &BTreeMap<UnitId, UnitLiveFacts>,
) -> Result<super::LiveByteBound, OwnershipError> {
    let mut remaining = units.iter().map(|unit| unit.id).collect::<BTreeSet<_>>();
    let mut components = Vec::<BTreeSet<UnitId>>::new();
    while let Some(first) = remaining.first().copied() {
        let component = remaining
            .iter()
            .copied()
            .filter(|candidate| {
                unit_reaches(first, *candidate, facts) && unit_reaches(*candidate, first, facts)
            })
            .collect::<BTreeSet<_>>();
        for unit in &component {
            remaining.remove(unit);
        }
        components.push(component);
    }
    let component_of = components
        .iter()
        .enumerate()
        .flat_map(|(component, units)| units.iter().map(move |unit| (*unit, component)))
        .collect::<BTreeMap<_, _>>();
    let mut composed = components
        .iter()
        .map(|_| ComponentLiveFacts {
            local: super::LiveByteBound::Exact(0),
            outgoing: Vec::new(),
        })
        .collect::<Vec<_>>();
    for unit in units {
        let component = component_of[&unit.id];
        let unit_facts = &facts[&unit.id];
        composed[component].local = composed[component]
            .local
            .maximum(unit_facts.local_max_live_bytes);
        for call in &unit_facts.direct_calls {
            let callee_component = component_of[&call.callee];
            if callee_component == component {
                composed[component].local = composed[component].local.maximum(match call.carry {
                    super::LiveByteBound::Exact(0) => super::LiveByteBound::Exact(0),
                    super::LiveByteBound::Exact(_) | super::LiveByteBound::Unbounded => {
                        super::LiveByteBound::Unbounded
                    }
                    super::LiveByteBound::Unknown => super::LiveByteBound::Unknown,
                });
            } else {
                composed[component]
                    .outgoing
                    .push((callee_component, call.carry));
            }
        }
    }

    fn component_bound(
        index: usize,
        facts: &[ComponentLiveFacts],
        memo: &mut [Option<super::LiveByteBound>],
        visiting: &mut BTreeSet<usize>,
    ) -> Result<super::LiveByteBound, OwnershipError> {
        if let Some(bound) = memo[index] {
            return Ok(bound);
        }
        if !visiting.insert(index) {
            return Err(OwnershipError::LoweringInvariant {
                unit: "verified live-byte bound".to_string(),
                detail: "SCC condensation graph retained a cycle".to_string(),
            });
        }
        let mut bound = facts[index].local;
        for (callee, carry) in &facts[index].outgoing {
            let callee = component_bound(*callee, facts, memo, visiting)?;
            bound = bound.maximum(add_live_byte_bounds(*carry, callee, || {
                "composing caller carry with callee peak".to_string()
            })?);
        }
        visiting.remove(&index);
        memo[index] = Some(bound);
        Ok(bound)
    }

    let mut memo = vec![None; composed.len()];
    let mut result = super::LiveByteBound::Exact(0);
    for index in 0..composed.len() {
        result = result.maximum(component_bound(
            index,
            &composed,
            &mut memo,
            &mut BTreeSet::new(),
        )?);
    }
    Ok(result)
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
        for owner in block.params.iter().map(|p| p.owner).chain(
            block
                .ops
                .iter()
                .filter_map(|operation| destination(&operation.kind)),
        ) {
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
        Op::Project { .. } | Op::Drop { .. } | Op::Discard { .. } | Op::RootConsume { .. } => None,
    }
}

fn classify_tail_calls(unit: &Unit) -> Vec<(UnitId, OpId)> {
    if unit.kind != UnitKind::Function {
        return Vec::new();
    }
    unit.blocks
        .iter()
        .flat_map(|block| &block.ops)
        .filter_map(|operation| match &operation.kind {
            Op::Apply {
                dest: Some(owner),
                kind: ApplyKind::DirectCall { .. },
                ..
            } if owner_reaches_return(unit, *owner, &mut BTreeSet::new()) => {
                Some((unit.id, operation.id))
            }
            _ => None,
        })
        .collect()
}

/// Recognize only ownership-transparent result chains. This is deliberately
/// independent of the Apply label and of lowering's lexical `tail` hint.
fn owner_reaches_return(unit: &Unit, owner: OwnerId, visiting: &mut BTreeSet<OwnerId>) -> bool {
    if !visiting.insert(owner) {
        return false;
    }
    let mut reaches_terminal = false;
    for block in &unit.blocks {
        for operation in &block.ops {
            match &operation.kind {
                Op::Project { source } if source.owner == owner => {
                    if source.use_ != OwnershipUse::Borrow {
                        visiting.remove(&owner);
                        return false;
                    }
                }
                Op::Apply { args, .. } if args.iter().any(|arg| arg.owner == owner) => {
                    visiting.remove(&owner);
                    return false;
                }
                Op::Copy { source, .. } if source.owner == owner => {
                    visiting.remove(&owner);
                    return false;
                }
                Op::LoopItem { list, .. } if list.owner == owner => {
                    visiting.remove(&owner);
                    return false;
                }
                Op::Drop { owner: operand } if operand.owner == owner => {
                    visiting.remove(&owner);
                    return false;
                }
                Op::Discard { owner: discarded } if *discarded == owner => {
                    visiting.remove(&owner);
                    return false;
                }
                Op::RootConsume { owner: operand, .. } if operand.owner == owner => {
                    visiting.remove(&owner);
                    return false;
                }
                Op::Define { .. }
                | Op::Apply { .. }
                | Op::Copy { .. }
                | Op::Project { .. }
                | Op::LoopItem { .. }
                | Op::Drop { .. }
                | Op::Discard { .. }
                | Op::RootConsume { .. } => {}
            }
        }
        match &block.terminator {
            Terminator::Return { result } if result.owner == owner => {
                if result.use_ != OwnershipUse::Move {
                    visiting.remove(&owner);
                    return false;
                }
                reaches_terminal = true;
            }
            Terminator::Branch { condition, .. } if condition.owner == owner => {
                visiting.remove(&owner);
                return false;
            }
            Terminator::Match { scrutinee, .. } if scrutinee.owner == owner => {
                visiting.remove(&owner);
                return false;
            }
            Terminator::Loop { list, .. } if list.owner == owner => {
                visiting.remove(&owner);
                return false;
            }
            Terminator::Return { .. }
            | Terminator::Jump(_)
            | Terminator::Branch { .. }
            | Terminator::Match { .. }
            | Terminator::Loop { .. }
            | Terminator::Exit => {}
        }
        for edge in terminator_edges(&block.terminator) {
            if edge
                .terminals
                .iter()
                .any(|terminal| terminal.kind.owner() == owner)
            {
                visiting.remove(&owner);
                return false;
            }
            for (arg, param) in edge.args.iter().zip(
                unit.blocks
                    .iter()
                    .find(|candidate| candidate.id == edge.target)
                    .into_iter()
                    .flat_map(|target| &target.params),
            ) {
                if arg.owner != owner {
                    continue;
                }
                if arg.use_ != OwnershipUse::Move
                    || param.mode != ParamMode::Owned
                    || !owner_reaches_return(unit, param.owner, visiting)
                {
                    visiting.remove(&owner);
                    return false;
                }
                reaches_terminal = true;
            }
        }
    }
    visiting.remove(&owner);
    reaches_terminal
}

fn terminator_edges(terminator: &Terminator) -> Vec<&Edge> {
    match terminator {
        Terminator::Return { .. } | Terminator::Exit => Vec::new(),
        Terminator::Jump(edge) => vec![edge],
        Terminator::Branch {
            then_edge,
            else_edge,
            ..
        }
        | Terminator::Loop {
            body_edge: then_edge,
            exit_edge: else_edge,
            ..
        } => vec![then_edge, else_edge],
        Terminator::Match { arms, .. } => arms.iter().collect(),
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
    let reached = reachable_blocks(unit, blocks)?;
    if let Some(block) = unit.blocks.iter().find(|b| !reached.contains(&b.id)) {
        return Err(OwnershipError::UnreachableBlock {
            unit: unit.name.clone(),
            block: block.id.0,
        });
    }
    Ok(())
}

fn reachable_blocks(
    unit: &Unit,
    blocks: &BTreeMap<BlockId, &Block>,
) -> Result<BTreeSet<BlockId>, OwnershipError> {
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
    Ok(reached)
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
    units: &BTreeMap<UnitId, &Unit>,
    callable_schemas: &BTreeMap<UnitId, CallableSchema>,
    live: &mut BTreeSet<OwnerId>,
) -> Result<(), OwnershipError> {
    match op {
        Op::Define { dest, .. } => define(unit, *dest, live),
        Op::Apply {
            dest,
            label,
            kind,
            schema,
            args,
        } => {
            let direct_schema = if let ApplyKind::DirectCall { callee } = kind {
                match units.get(callee) {
                    None => {
                        return Err(OwnershipError::MissingUnit {
                            caller: unit.name.clone(),
                            unit: callee.0,
                        });
                    }
                    Some(target) if target.kind != UnitKind::Function => {
                        return Err(OwnershipError::NonFunctionCallee {
                            caller: unit.name.clone(),
                            unit: callee.0,
                        });
                    }
                    Some(_)
                        if callable_schemas.get(callee).map(|schema| &schema.operation)
                            != Some(schema) =>
                    {
                        return Err(OwnershipError::DirectCallSchema {
                            caller: unit.name.clone(),
                            unit: callee.0,
                        });
                    }
                    Some(_) => callable_schemas.get(callee),
                }
            } else {
                None
            };
            if args.len() != schema.operands.len() {
                return Err(OwnershipError::OperationArity {
                    unit: unit.name.clone(),
                    block: block.id.0,
                    label: label.clone(),
                    expected: schema.operands.len(),
                    actual: args.len(),
                });
            }
            if let Some(direct_schema) = direct_schema {
                for (argument, (arg, expected)) in args
                    .iter()
                    .zip(&direct_schema.parameter_classes)
                    .enumerate()
                {
                    let actual = unit
                        .owners
                        .get(&arg.owner)
                        .ok_or_else(|| incomplete(unit, arg.owner, "metadata"))?
                        .class;
                    if actual != *expected {
                        let ApplyKind::DirectCall { callee } = kind else {
                            unreachable!("derived direct-call schema requires a direct call")
                        };
                        return Err(OwnershipError::DirectCallArgumentClass {
                            caller: unit.name.clone(),
                            unit: callee.0,
                            argument,
                            expected: expected.to_string(),
                            actual: actual.to_string(),
                        });
                    }
                }
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
        Op::Discard { owner } => {
            if !definitions.contains(owner) {
                return Err(incomplete(unit, *owner, "definition"));
            }
            if unit.owners[owner].class.is_heap() {
                return Err(OwnershipError::HeapDiscard {
                    unit: unit.name.clone(),
                    owner: owner.0,
                    block: block.id.0,
                });
            }
            use_operand(
                unit,
                block.id,
                &Operand::move_(*owner),
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
    for terminal in &edge.terminals {
        verify_edge_terminal(unit, block.id, terminal.kind, definitions, &mut next)?;
    }
    check_borrows(unit, target.id, &next)?;
    if let Some(expected) = incoming.get(&target.id) {
        if expected != &next {
            return Err(OwnershipError::JoinMismatch {
                unit: unit.name.clone(),
                block: target.id.0,
                only_here: owner_difference(unit, &next, expected),
                only_earlier: owner_difference(unit, expected, &next),
                here_count: next.len(),
                earlier_count: expected.len(),
            });
        }
    } else {
        incoming.insert(target.id, next);
        queue.push_back(target.id);
    }
    Ok(())
}

fn verify_edge_terminal(
    unit: &Unit,
    block: BlockId,
    terminal: Terminal,
    definitions: &BTreeSet<OwnerId>,
    live: &mut BTreeSet<OwnerId>,
) -> Result<(), OwnershipError> {
    let owner = terminal.owner();
    if !definitions.contains(&owner) {
        return Err(incomplete(unit, owner, "definition"));
    }
    match terminal {
        Terminal::Drop(_) if !unit.owners[&owner].class.is_heap() => {
            return Err(OwnershipError::NonHeapDrop {
                unit: unit.name.clone(),
                owner: owner.0,
                block: block.0,
            });
        }
        Terminal::Discard(_) if unit.owners[&owner].class.is_heap() => {
            return Err(OwnershipError::HeapDiscard {
                unit: unit.name.clone(),
                owner: owner.0,
                block: block.0,
            });
        }
        Terminal::Drop(_) | Terminal::Discard(_) => {}
    }
    use_operand(
        unit,
        block,
        &Operand::move_(owner),
        Some(OwnershipUse::Move),
        definitions,
        live,
    )
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

/// The owners live in `from` and not in `to`, labelled with their source
/// binding names. A join mismatch is only actionable if the reader can see
/// which owner the two paths disagree about (chelis#2122).
fn owner_difference(unit: &Unit, from: &BTreeSet<OwnerId>, to: &BTreeSet<OwnerId>) -> String {
    let labels = from
        .difference(to)
        .map(|owner| crate::ownership::render::owner_label(unit, *owner))
        .collect::<Vec<_>>();
    if labels.is_empty() {
        "none".to_string()
    } else {
        labels.join(", ")
    }
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
