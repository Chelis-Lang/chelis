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

use std::collections::BTreeMap;

use chelis_types::manifest::{ManifestedProgram, RootManifest};

use crate::dag::{Dag, DagNode, NodeId, RiscOp, SymbolicDimBinding, SymbolicDimOccurrence};
use crate::host::{
    ConcreteHostBinding, ConcreteHostExpr, ConcreteHostFunction, ConcreteHostParam,
    ConcreteHostProgram, HostFunctionOrigin, HostFunctionSpecialization, HostTensorHelper,
    HostTensorInput, HostTensorSpecialization, SummaryRejection,
};

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
pub use ir::{HostSiteId, HostSiteKind};

/// Immutable cursor over the exact verified DAG payload. The raw [`Dag`]
/// remains private so a backend can inspect only the payload whose ownership
/// plan was verified, without recovering an unchecked sibling graph.
#[derive(Clone, Copy)]
pub struct VerifiedDagView<'a> {
    dag: &'a Dag,
}

impl<'a> VerifiedDagView<'a> {
    pub fn nodes(self) -> &'a [DagNode] {
        self.dag.nodes()
    }

    pub fn roots(self) -> &'a [NodeId] {
        self.dag.roots()
    }

    pub fn get(self, id: NodeId) -> Option<&'a DagNode> {
        self.dag.get(id)
    }

    pub fn len(self) -> usize {
        self.dag.len()
    }

    pub fn is_empty(self) -> bool {
        self.dag.is_empty()
    }

    pub fn is_root(self, id: NodeId) -> bool {
        self.dag.is_root(id)
    }

    pub fn topological_order(self) -> Vec<NodeId> {
        self.dag.topological_order()
    }

    pub fn symbolic_bindings(self) -> Vec<SymbolicDimBinding> {
        crate::dag::symbolic_bindings(self.dag)
    }

    pub fn symbolic_occurrences(self) -> Vec<SymbolicDimOccurrence> {
        crate::dag::symbolic_occurrences(self.dag)
    }

    pub fn symbolic_params(self) -> Vec<String> {
        crate::dag::symbolic_params(self.dag)
    }

    pub fn reduction_inlined_fused_elems(self) -> chelis_unord::UnordSet<NodeId> {
        crate::fuse::reduction_inlined_fused_elems(self.dag)
    }

    pub fn first_integer_abs_node(self) -> Option<NodeId> {
        crate::analysis::first_integer_abs_node(self.dag)
    }

    pub fn first_fused_integer_abs_node(self) -> Option<NodeId> {
        crate::analysis::first_fused_integer_abs_node(self.dag)
    }

    pub fn check_axis_sources(
        self,
        stage: chelis_types::unsupported::Stage,
    ) -> Result<(), chelis_types::unsupported::Unsupported> {
        crate::axis_sources::check_axis_sources(self.dag, stage)
    }
}

/// One immutable record in the verified host payload/site bijection.
#[derive(Clone, Copy)]
pub struct VerifiedHostSiteView<'a> {
    record: &'a ir::HostSiteRecord,
}

/// Closed classification for a verified ownership directive attached to a
/// host-emission site. The verifier's block, operation, and owner identities
/// remain private; consumers can branch only on this typed semantic role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifiedHostSiteActionKind {
    Operation,
    Terminator,
    ControlEdge,
    ManifestRoot,
}

impl<'a> VerifiedHostSiteView<'a> {
    pub fn id(self) -> HostSiteId {
        self.record.id
    }

    pub fn kind(self) -> HostSiteKind {
        self.record.kind
    }

    pub fn actions(self) -> impl ExactSizeIterator<Item = VerifiedHostSiteActionKind> + 'a {
        self.record.actions.iter().map(|action| match action {
            ir::HostSiteAction::Operation { .. } => VerifiedHostSiteActionKind::Operation,
            ir::HostSiteAction::Terminator { .. } => VerifiedHostSiteActionKind::Terminator,
            ir::HostSiteAction::ControlEdge { .. } => VerifiedHostSiteActionKind::ControlEdge,
            ir::HostSiteAction::Root { .. } => VerifiedHostSiteActionKind::ManifestRoot,
        })
    }
}

/// Immutable cursor over one nested host tensor helper and its verified DAG.
#[derive(Clone, Copy)]
pub struct VerifiedHostTensorHelperView<'a> {
    helper: &'a HostTensorHelper,
    dag: VerifiedDagView<'a>,
}

impl<'a> VerifiedHostTensorHelperView<'a> {
    pub fn name(self) -> &'a str {
        &self.helper.name
    }

    pub fn inputs(self) -> &'a [HostTensorInput] {
        &self.helper.inputs
    }

    pub fn output(self) -> &'a crate::dag::TensorType {
        &self.helper.output
    }

    pub fn specialization(self) -> Option<&'a HostTensorSpecialization> {
        self.helper.specialization.as_ref()
    }

    pub fn summary_rejection(self) -> Option<&'a crate::host::HelperSummaryRejection> {
        self.helper.summary_rejection.as_ref()
    }

    pub fn dag(self) -> VerifiedDagView<'a> {
        self.dag
    }
}

/// Immutable cursor over one concrete host function and its verified helpers.
#[derive(Clone, Copy)]
pub struct VerifiedHostFunctionView<'a> {
    emission: VerifiedHostEmission<'a>,
    index: usize,
}

impl<'a> VerifiedHostFunctionView<'a> {
    fn function(self) -> &'a ConcreteHostFunction {
        &self.emission.payload.program.functions[self.index]
    }

    pub fn name(self) -> &'a str {
        &self.function().name
    }

    pub fn params(self) -> &'a [ConcreteHostParam] {
        &self.function().params
    }

    pub fn ret_ty(self) -> &'a crate::host_type_state::ConcreteHostType {
        &self.function().ret_ty
    }

    pub fn body(self) -> &'a ConcreteHostExpr {
        &self.function().body
    }

    pub fn origin(self) -> HostFunctionOrigin {
        self.function().origin
    }

    pub fn specialization(self) -> Option<&'a HostFunctionSpecialization> {
        self.function().specialization.as_ref()
    }

    pub fn summary_rejections(self) -> &'a [SummaryRejection] {
        &self.function().summary_rejections
    }

    pub fn tensor_helper_count(self) -> usize {
        self.function().tensor_helpers.len()
    }

    pub fn tensor_helper(self, helper: usize) -> Option<VerifiedHostTensorHelperView<'a>> {
        let raw = self.function().tensor_helpers.get(helper)?;
        let _plan = self.emission.nested_dags.iter().find(|candidate| {
            candidate.location
                == NestedDagLocation::Function {
                    function: self.index,
                    helper,
                }
        })?;
        Some(VerifiedHostTensorHelperView {
            helper: raw,
            dag: VerifiedDagView { dag: &raw.dag },
        })
    }
}

/// Immutable view of the exact verified host emission payload. Its fields are
/// private; nested DAGs are reachable only as verified child cursors.
#[derive(Clone, Copy)]
pub struct VerifiedHostEmission<'a> {
    payload: &'a HostEmissionPayload,
    sites: &'a ir::HostSiteMap,
    nested_dags: &'a [NestedDagProof],
}

impl<'a> VerifiedHostEmission<'a> {
    pub fn globals(self) -> &'a [ConcreteHostBinding] {
        &self.payload.program.globals
    }

    pub fn function_count(self) -> usize {
        self.payload.program.functions.len()
    }

    pub fn function(self, index: usize) -> Option<VerifiedHostFunctionView<'a>> {
        (index < self.function_count()).then_some(VerifiedHostFunctionView {
            emission: self,
            index,
        })
    }

    pub fn global_tensor_helper_count(self) -> usize {
        self.payload.program.global_tensor_helpers.len()
    }

    pub fn global_tensor_helper(self, helper: usize) -> Option<VerifiedHostTensorHelperView<'a>> {
        let raw = self.payload.program.global_tensor_helpers.get(helper)?;
        self.nested_dags
            .iter()
            .find(|candidate| candidate.location == NestedDagLocation::Global(helper))?;
        Some(VerifiedHostTensorHelperView {
            helper: raw,
            dag: VerifiedDagView { dag: &raw.dag },
        })
    }

    pub fn summary_rejections(self) -> &'a [SummaryRejection] {
        &self.payload.program.summary_rejections
    }

    pub fn manifest(self) -> &'a RootManifest {
        &self.payload.manifest
    }

    pub fn sites(self) -> impl ExactSizeIterator<Item = VerifiedHostSiteView<'a>> + 'a {
        self.sites
            .records
            .iter()
            .map(|record| VerifiedHostSiteView { record })
    }
}

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
            let payload = host_payload(&program)?;
            verify::verify_host_sites(&payload.program, &payload.manifest, ir_program, sites)?;
            verify_manifest_sinks(payload, ir_program)?;
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
    pub fn emission(&self) -> VerifiedHostEmission<'_> {
        let OwnershipProof::Host {
            sites, nested_dags, ..
        } = &self.0.proof
        else {
            unreachable!("sealed host specialization")
        };
        let payload = (&self.0.payload as &dyn std::any::Any)
            .downcast_ref::<HostEmissionPayload>()
            .expect("sealed host payload specialization");
        VerifiedHostEmission {
            payload,
            sites,
            nested_dags,
        }
    }

    pub fn render(&self) -> String {
        match &self.0.proof {
            OwnershipProof::Host { program, .. } => render::render(program),
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
    pub fn emission(&self) -> VerifiedDagView<'_> {
        let payload = (&self.0.payload as &dyn std::any::Any)
            .downcast_ref::<DagEmissionPayload>()
            .expect("sealed DAG payload specialization");
        VerifiedDagView { dag: &payload.dag }
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DagOwnerState {
    BorrowedLive,
    OwnedLive,
    OwnedTerminal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DagDirective {
    BorrowLoad { node: NodeId },
    Produce { node: NodeId, borrows: Vec<NodeId> },
    Clone { node: NodeId, source: NodeId },
    MoveProduce { node: NodeId, source: NodeId },
    Store { node: NodeId, source: NodeId },
    StoreRoot { node: NodeId },
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
                    consume_dag_owner(&mut states, node.inputs[0], node.id)?;
                    directives.push(DagDirective::Drop {
                        node: node.id,
                        source: node.inputs[0],
                    });
                }
                RiscOp::Realize => {
                    require_dag_arity(node.id, "realize", 1, node.inputs.len())?;
                    consume_dag_owner(&mut states, node.inputs[0], node.id)?;
                    owners.insert(node.id, DagOwnerOrigin::OwnedProducer);
                    states.insert(node.id, DagOwnerState::OwnedLive);
                    directives.push(DagDirective::MoveProduce {
                        node: node.id,
                        source: node.inputs[0],
                    });
                }
                RiscOp::Store { .. } => {
                    require_dag_arity(node.id, "store", 1, node.inputs.len())?;
                    consume_dag_owner(&mut states, node.inputs[0], node.id)?;
                    directives.push(DagDirective::Store {
                        node: node.id,
                        source: node.inputs[0],
                    });
                }
                _ => {
                    let mut borrows = node.inputs.clone();
                    for dependency in &node.shape_deps {
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

    fn verify(&self, dag: &Dag) -> Result<(), OwnershipError> {
        let structural_errors = crate::verify::verify(dag);
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
                    consume_dag_owner(&mut states, node.inputs[0], node.id)?;
                    require_exact_dag_directive(
                        directive,
                        &DagDirective::Drop {
                            node: node.id,
                            source: node.inputs[0],
                        },
                        node.id,
                    )?;
                }
                RiscOp::Realize => {
                    require_dag_arity(node.id, "realize", 1, node.inputs.len())?;
                    consume_dag_owner(&mut states, node.inputs[0], node.id)?;
                    require_exact_dag_directive(
                        directive,
                        &DagDirective::MoveProduce {
                            node: node.id,
                            source: node.inputs[0],
                        },
                        node.id,
                    )?;
                    owners.insert(node.id, DagOwnerOrigin::OwnedProducer);
                    states.insert(node.id, DagOwnerState::OwnedLive);
                }
                RiscOp::Store { .. } => {
                    require_dag_arity(node.id, "store", 1, node.inputs.len())?;
                    consume_dag_owner(&mut states, node.inputs[0], node.id)?;
                    require_exact_dag_directive(
                        directive,
                        &DagDirective::Store {
                            node: node.id,
                            source: node.inputs[0],
                        },
                        node.id,
                    )?;
                }
                _ => {
                    let mut borrows = node.inputs.clone();
                    for dependency in &node.shape_deps {
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
                DagDirective::StoreRoot { node } => {
                    let _ = writeln!(out, "store-root n{}", node.0);
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

fn validate_dag_dependencies(dag: &Dag, node: &crate::dag::DagNode) -> Result<(), OwnershipError> {
    for input in node.inputs.iter().chain(&node.shape_deps) {
        if input.0 >= node.id.0 || dag.get(*input).is_none() {
            return Err(OwnershipError::DagInput {
                node: node.id.0,
                input: input.0,
            });
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests;
