//! Sealed ownership boundary for compiled-value-ownership Phase 2.
//!
//! Ownership lowering consumes the exact post-selection emission payload.
//! [`verify_ownership`] is the only transition to a backend-admissible value;
//! neither payload specialization exposes an owned or mutable raw payload.
//!
//! `HostSiteId` is opaque and structurally allocated. It is not a public
//! integer carrier and cannot be constructed from an identifier spelling.
//!
//! ```compile_fail
//! use chelis_ir::ownership::{HostEmissionPayload, OwnershipProgram};
//! fn bypass(payload: HostEmissionPayload) -> OwnershipProgram<HostEmissionPayload> {
//!     OwnershipProgram { payload, proof: unreachable!() }
//! }
//! ```
//!
//! ```compile_fail
//! use chelis_ir::ownership::{HostEmissionPayload, OwnershipProgram,
//!     VerifiedOwnershipProgram};
//! fn bypass(p: OwnershipProgram<HostEmissionPayload>)
//!     -> VerifiedOwnershipProgram<HostEmissionPayload>
//! {
//!     VerifiedOwnershipProgram(p)
//! }
//! ```
//!
//! ```compile_fail
//! use chelis_ir::ownership::{HostEmissionPayload, OwnershipProgram};
//! fn destructure(p: OwnershipProgram<HostEmissionPayload>) -> HostEmissionPayload {
//!     let OwnershipProgram { payload, .. } = p;
//!     payload
//! }
//! ```
//!
//! Neither concrete emission payload exposes a public constructor:
//!
//! ```compile_fail
//! # use chelis_ir::host::ConcreteHostProgram;
//! # use chelis_types::manifest::RootManifest;
//! use chelis_ir::ownership::HostEmissionPayload;
//! fn forge(program: ConcreteHostProgram, manifest: RootManifest) -> HostEmissionPayload {
//!     HostEmissionPayload { program, manifest }
//! }
//! ```
//!
//! ```compile_fail
//! # use chelis_ir::dag::Dag;
//! use chelis_ir::ownership::DagEmissionPayload;
//! fn forge(dag: Dag) -> DagEmissionPayload { DagEmissionPayload { dag } }
//! ```
//!
//! A lowering call consumes the host program, so no mutable raw sibling can
//! survive the boundary:
//!
//! ```compile_fail
//! # use chelis_ir::host::ConcreteHostProgram;
//! # use chelis_types::manifest::ManifestedProgram;
//! use chelis_ir::ownership::lower_host_ownership;
//! fn mutate_after_lowering(manifested: &ManifestedProgram, mut host: ConcreteHostProgram) {
//!     let _lowered = lower_host_ownership(manifested, host).unwrap();
//!     host.globals.reverse();
//! }
//! ```
//!
//! A verified payload cannot be extracted for mutation or reordered:
//!
//! ```compile_fail
//! # use chelis_ir::ownership::VerifiedHostProgram;
//! fn extract(mut verified: VerifiedHostProgram) {
//!     verified.payload.program.globals.reverse();
//! }
//! ```
//!
//! Host-site identities cannot be forged from a public integer carrier:
//!
//! ```compile_fail
//! use chelis_ir::ownership::HostSiteId;
//! fn forge() -> HostSiteId { HostSiteId::from_index(7) }
//! ```

use std::collections::{BTreeMap, BTreeSet};

use chelis_types::manifest::{ManifestedProgram, RootManifest};

use crate::dag::{Dag, NodeId, RiscOp};
use crate::host::ConcreteHostProgram;

#[expect(
    dead_code,
    reason = "the closed heap census includes planner-only TensorStorage"
)]
mod classify;
mod error;
mod ir;
mod lower;
mod render;
mod verify;

pub use error::OwnershipError;
pub use ir::HostSiteId;

/// Closed payload identity. Public only so the sealed generic verifier can
/// carry a public bound without exposing its private representation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadKind {
    Host,
    Dag,
}

mod sealed {
    pub trait Sealed {}
}

/// A sealed emission-payload specialization. Downstream crates can name the
/// concrete host and DAG payloads but cannot add a third unchecked lane.
pub trait EmissionPayload: sealed::Sealed + 'static {
    #[doc(hidden)]
    const KIND: PayloadKind;
}

/// Exact host payload after manifest observation materialization.
#[derive(Debug)]
pub struct HostEmissionPayload {
    program: ConcreteHostProgram,
    manifest: RootManifest,
}

impl sealed::Sealed for HostEmissionPayload {}
impl EmissionPayload for HostEmissionPayload {
    const KIND: PayloadKind = PayloadKind::Host;
}

/// Exact post-optimization standalone tensor DAG.
#[derive(Debug)]
pub struct DagEmissionPayload {
    dag: Dag,
}

impl sealed::Sealed for DagEmissionPayload {}
impl EmissionPayload for DagEmissionPayload {
    const KIND: PayloadKind = PayloadKind::Dag;
}

#[derive(Debug)]
enum OwnershipProof {
    Host {
        program: ir::OwnershipProgram,
        sites: ir::HostSiteMap,
        nested_dags: Vec<NestedDagProof>,
    },
    Dag(DagOwnershipPlan),
}

#[derive(Debug)]
struct NestedDagProof {
    location: NestedDagLocation,
    plan: DagOwnershipPlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NestedDagLocation {
    Global(usize),
    Function { function: usize, helper: usize },
}

/// Unverified ownership-lowered form. Its fields and constructor are private.
#[derive(Debug)]
pub struct OwnershipProgram<P: EmissionPayload> {
    payload: P,
    proof: OwnershipProof,
}

/// Verified host or DAG payload. Only [`verify_ownership`] constructs it.
#[derive(Debug)]
pub struct VerifiedOwnershipProgram<P: EmissionPayload>(OwnershipProgram<P>);

pub type HostOwnershipProgram = OwnershipProgram<HostEmissionPayload>;
pub type DagOwnershipProgram = OwnershipProgram<DagEmissionPayload>;
pub type VerifiedHostProgram = VerifiedOwnershipProgram<HostEmissionPayload>;
pub type VerifiedDagProgram = VerifiedOwnershipProgram<DagEmissionPayload>;

/// Consume a concrete host program, materialize every manifested host root,
/// and lower the exact resulting payload to ownership IR.
pub fn lower_host_ownership(
    manifested: &ManifestedProgram,
    mut host: ConcreteHostProgram,
) -> Result<HostOwnershipProgram, OwnershipError> {
    let root_bindings = lower::materialize_manifest_roots(&mut host, manifested.manifest())?;
    let mut sites = ir::HostSiteBuilder::default();
    let program = lower::lower(
        manifested.checked(),
        &host,
        manifested.manifest(),
        &root_bindings,
        &mut sites,
    )?;
    let nested_dags = lower_nested_dags(&host)?;
    Ok(OwnershipProgram {
        payload: HostEmissionPayload {
            program: host,
            manifest: manifested.manifest().clone(),
        },
        proof: OwnershipProof::Host {
            program,
            sites: sites.finish(),
            nested_dags,
        },
    })
}

/// Consume a standalone DAG and attach an explicit ownership directive for
/// every node, root, and implicit scope-exit drop.
pub fn lower_dag_ownership(dag: Dag) -> Result<DagOwnershipProgram, OwnershipError> {
    let plan = DagOwnershipPlan::lower(&dag)?;
    Ok(OwnershipProgram {
        payload: DagEmissionPayload { dag },
        proof: OwnershipProof::Dag(plan),
    })
}

/// Verify one sealed specialization and consume the unverified carrier.
pub fn verify_ownership<P: EmissionPayload>(
    program: OwnershipProgram<P>,
) -> Result<VerifiedOwnershipProgram<P>, OwnershipError> {
    match (&program.proof, P::KIND) {
        (
            OwnershipProof::Host {
                program: ir_program,
                sites,
                nested_dags,
            },
            PayloadKind::Host,
        ) => {
            verify::verify(ir_program)?;
            verify::verify_host_sites(ir_program, sites)?;
            verify_manifest_sinks(host_payload(&program)?, ir_program)?;
            verify_nested_dags(&program, nested_dags)?;
        }
        (OwnershipProof::Dag(plan), PayloadKind::Dag) => {
            let dag = dag_payload(&program)?;
            plan.verify(dag)?;
        }
        _ => {
            return Err(OwnershipError::LoweringInvariant {
                unit: "ownership-boundary".to_string(),
                detail: "payload specialization does not match its private proof".to_string(),
            });
        }
    }
    Ok(VerifiedOwnershipProgram(program))
}

fn verify_manifest_sinks(
    payload: &HostEmissionPayload,
    program: &ir::OwnershipProgram,
) -> Result<(), OwnershipError> {
    use chelis_types::types::Lane;
    let expected = payload
        .manifest
        .entries
        .iter()
        .filter(|entry| entry.lane == Lane::Host)
        .map(|entry| entry.name.as_str())
        .collect::<Vec<_>>();
    let actual = program
        .units
        .iter()
        .find(|unit| unit.kind == ir::UnitKind::Roots)
        .into_iter()
        .flat_map(|unit| &unit.blocks)
        .flat_map(|block| &block.ops)
        .filter_map(|op| match op {
            ir::Op::RootConsume { root, .. } => Some(root.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if expected == actual {
        Ok(())
    } else {
        Err(OwnershipError::LoweringInvariant {
            unit: "roots".to_string(),
            detail: format!(
                "manifest sink order diverged from exact payload: expected {expected:?}, got {actual:?}"
            ),
        })
    }
}

impl VerifiedHostProgram {
    pub fn render(&self) -> String {
        match &self.0.proof {
            OwnershipProof::Host { program, .. } => render::render(program),
            OwnershipProof::Dag(_) => unreachable!("sealed host specialization"),
        }
    }

    pub fn host_site_count(&self) -> usize {
        match &self.0.proof {
            OwnershipProof::Host { sites, .. } => sites.records.len(),
            OwnershipProof::Dag(_) => unreachable!("sealed host specialization"),
        }
    }

    pub fn host_directive_site_count(&self) -> usize {
        match &self.0.proof {
            OwnershipProof::Host { sites, .. } => sites
                .records
                .iter()
                .filter(|site| !site.actions.is_empty())
                .count(),
            OwnershipProof::Dag(_) => unreachable!("sealed host specialization"),
        }
    }

    pub fn nested_dag_count(&self) -> usize {
        match &self.0.proof {
            OwnershipProof::Host { nested_dags, .. } => nested_dags.len(),
            OwnershipProof::Dag(_) => unreachable!("sealed host specialization"),
        }
    }

    pub fn nested_dag_render(&self, index: usize) -> Option<String> {
        let OwnershipProof::Host { nested_dags, .. } = &self.0.proof else {
            unreachable!("sealed host specialization")
        };
        nested_dags.get(index).map(|child| child.plan.render())
    }
}

impl VerifiedDagProgram {
    pub fn render(&self) -> String {
        match &self.0.proof {
            OwnershipProof::Dag(plan) => plan.render(),
            OwnershipProof::Host { .. } => unreachable!("sealed DAG specialization"),
        }
    }
}

fn dag_payload<P: EmissionPayload>(program: &OwnershipProgram<P>) -> Result<&Dag, OwnershipError> {
    let payload = (&program.payload as &dyn std::any::Any)
        .downcast_ref::<DagEmissionPayload>()
        .ok_or_else(|| OwnershipError::LoweringInvariant {
            unit: "ownership-boundary".to_string(),
            detail: "requested DAG payload from host specialization".to_string(),
        })?;
    Ok(&payload.dag)
}

fn host_payload<P: EmissionPayload>(
    program: &OwnershipProgram<P>,
) -> Result<&HostEmissionPayload, OwnershipError> {
    (&program.payload as &dyn std::any::Any)
        .downcast_ref::<HostEmissionPayload>()
        .ok_or_else(|| OwnershipError::LoweringInvariant {
            unit: "ownership-boundary".to_string(),
            detail: "requested host payload from DAG specialization".to_string(),
        })
}

fn lower_nested_dags(host: &ConcreteHostProgram) -> Result<Vec<NestedDagProof>, OwnershipError> {
    let mut result = Vec::new();
    for (helper, child) in host.global_tensor_helpers.iter().enumerate() {
        result.push(NestedDagProof {
            location: NestedDagLocation::Global(helper),
            plan: DagOwnershipPlan::lower(&child.dag)?,
        });
    }
    for (function, host_function) in host.functions.iter().enumerate() {
        for (helper, child) in host_function.tensor_helpers.iter().enumerate() {
            result.push(NestedDagProof {
                location: NestedDagLocation::Function { function, helper },
                plan: DagOwnershipPlan::lower(&child.dag)?,
            });
        }
    }
    Ok(result)
}

fn verify_nested_dags<P: EmissionPayload>(
    program: &OwnershipProgram<P>,
    children: &[NestedDagProof],
) -> Result<(), OwnershipError> {
    let host = host_payload(program)?;
    for child in children {
        let dag = match child.location {
            NestedDagLocation::Global(helper) => &host.program.global_tensor_helpers[helper].dag,
            NestedDagLocation::Function { function, helper } => {
                &host.program.functions[function].tensor_helpers[helper].dag
            }
        };
        child.plan.verify(dag)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DagOwnerOrigin {
    BorrowedLoad,
    OwnedProducer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DagDirective {
    BorrowLoad { node: NodeId },
    Produce { node: NodeId, borrows: Vec<NodeId> },
    Clone { node: NodeId, source: NodeId },
    MoveProduce { node: NodeId, source: NodeId },
    Store { node: NodeId, source: NodeId },
    Drop { node: NodeId, source: NodeId },
    Root { source: NodeId },
    RootClone { source: NodeId },
    ScopeDrop { source: NodeId },
}

#[derive(Debug)]
struct DagOwnershipPlan {
    owners: BTreeMap<NodeId, DagOwnerOrigin>,
    directives: Vec<DagDirective>,
}

impl DagOwnershipPlan {
    fn lower(dag: &Dag) -> Result<Self, OwnershipError> {
        let structural_errors = crate::verify::verify(dag);
        if !structural_errors.is_empty() {
            return Err(OwnershipError::LoweringInvariant {
                unit: "dag".to_string(),
                detail: structural_errors.join("; "),
            });
        }
        let mut owners = BTreeMap::new();
        let mut directives = Vec::new();
        let mut terminals = BTreeSet::new();
        for node in dag.nodes() {
            for input in &node.inputs {
                if input.0 >= node.id.0 || dag.get(*input).is_none() {
                    return Err(OwnershipError::DagInput {
                        node: node.id.0,
                        input: input.0,
                    });
                }
            }
            match &node.op {
                RiscOp::Load { .. } => {
                    require_dag_arity(node.id, "load", 0, node.inputs.len())?;
                    owners.insert(node.id, DagOwnerOrigin::BorrowedLoad);
                    directives.push(DagDirective::BorrowLoad { node: node.id });
                }
                RiscOp::Copy => {
                    require_dag_arity(node.id, "copy", 1, node.inputs.len())?;
                    owners.insert(node.id, DagOwnerOrigin::OwnedProducer);
                    directives.push(DagDirective::Clone {
                        node: node.id,
                        source: node.inputs[0],
                    });
                }
                RiscOp::Drop => {
                    require_dag_arity(node.id, "drop", 1, node.inputs.len())?;
                    record_dag_terminal(&mut terminals, node.inputs[0])?;
                    directives.push(DagDirective::Drop {
                        node: node.id,
                        source: node.inputs[0],
                    });
                }
                RiscOp::Realize => {
                    require_dag_arity(node.id, "realize", 1, node.inputs.len())?;
                    record_dag_terminal(&mut terminals, node.inputs[0])?;
                    owners.insert(node.id, DagOwnerOrigin::OwnedProducer);
                    directives.push(DagDirective::MoveProduce {
                        node: node.id,
                        source: node.inputs[0],
                    });
                }
                RiscOp::Store { .. } => {
                    require_dag_arity(node.id, "store", 1, node.inputs.len())?;
                    record_dag_terminal(&mut terminals, node.inputs[0])?;
                    directives.push(DagDirective::Store {
                        node: node.id,
                        source: node.inputs[0],
                    });
                }
                _ => {
                    owners.insert(node.id, DagOwnerOrigin::OwnedProducer);
                    let mut borrows = node.inputs.clone();
                    for dependency in &node.shape_deps {
                        if !borrows.contains(dependency) {
                            borrows.push(*dependency);
                        }
                    }
                    directives.push(DagDirective::Produce {
                        node: node.id,
                        borrows,
                    });
                }
            }
        }
        for root in dag.roots() {
            match owners.get(root) {
                Some(DagOwnerOrigin::OwnedProducer) => {
                    record_dag_terminal(&mut terminals, *root)?;
                    directives.push(DagDirective::Root { source: *root });
                }
                Some(DagOwnerOrigin::BorrowedLoad) => {
                    directives.push(DagDirective::RootClone { source: *root });
                }
                None => return Err(OwnershipError::DagInvalidRoot { root: root.0 }),
            }
        }
        for (&owner, origin) in &owners {
            if *origin == DagOwnerOrigin::OwnedProducer && !terminals.contains(&owner) {
                terminals.insert(owner);
                directives.push(DagDirective::ScopeDrop { source: owner });
            }
        }
        Ok(Self { owners, directives })
    }

    fn verify(&self, dag: &Dag) -> Result<(), OwnershipError> {
        let rebuilt = Self::lower(dag)?;
        if self.owners != rebuilt.owners || self.directives != rebuilt.directives {
            return Err(OwnershipError::LoweringInvariant {
                unit: "dag".to_string(),
                detail: "DAG payload and ownership directive map diverged".to_string(),
            });
        }
        Ok(())
    }

    fn render(&self) -> String {
        use std::fmt::Write as _;
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
                DagDirective::MoveProduce { node, source } => {
                    let _ = writeln!(out, "produce n{} move n{}", node.0, source.0);
                }
                DagDirective::Store { node, source } => {
                    let _ = writeln!(out, "store n{} move n{}", node.0, source.0);
                }
                DagDirective::Drop { node, source } => {
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

fn record_dag_terminal(
    terminals: &mut BTreeSet<NodeId>,
    owner: NodeId,
) -> Result<(), OwnershipError> {
    if terminals.insert(owner) {
        Ok(())
    } else {
        Err(OwnershipError::DagDuplicateTerminal { owner: owner.0 })
    }
}

#[cfg(test)]
mod tests;
