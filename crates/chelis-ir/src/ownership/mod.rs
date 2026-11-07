//! Sealed ownership boundary for compiled-value-ownership Phase 2.
//!
//! Ownership lowering consumes the exact post-selection emission payload.
//! [`verify_ownership`] is the only transition to a backend-admissible value.
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
//! # use chelis_ir_core::host::ConcreteHostProgram;
//! # use chelis_types::manifest::RootManifest;
//! use chelis_ir::ownership::HostEmissionPayload;
//! fn forge(program: ConcreteHostProgram, manifest: RootManifest) -> HostEmissionPayload {
//!     HostEmissionPayload { program, manifest }
//! }
//! ```
//!
//! ```compile_fail
//! use chelis_ir::ownership::DagEmissionPayload;
//! fn forge(dag: chelis_dag_core::dag::Dag) -> DagEmissionPayload { DagEmissionPayload { dag } }
//! ```
//!
//! A lowering call consumes the host program, so no mutable raw sibling can
//! survive the boundary:
//!
//! ```compile_fail
//! # use chelis_ir_core::host::ConcreteHostProgram;
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
//!
//! The live-set result is produced only by ownership verification. callers
//! cannot forge a smaller bound:
//!
//! ```compile_fail
//! use chelis_ir::ownership::VerifiedLiveSetBound;
//! fn forge() -> VerifiedLiveSetBound {
//!     VerifiedLiveSetBound { max_live_heap_owners: 0 }
//! }
//! ```
//!
//! Generic owner operands expose no binder spelling from which a backend
//! could reconstruct lifetime or alias state. Name projection is a distinct
//! capability carried only by the verified operations that bind payloads:
//!
//! ```compile_fail
//! use chelis_ir::ownership::VerifiedOwnerView;
//! fn reconstruct(owner: VerifiedOwnerView<'_>) -> bool {
//!     !owner.names().is_empty()
//! }
//! ```

use chelis_ir_core::ownership::dag::{DagAnalysisView, DagDirective};
use chelis_ir_core::ownership::{HostSiteId, HostSiteKind, LiveByteBound, OwnershipError};
use chelis_ir_core::ownership::{ir, last_use, locations, lower, render, verify};
use std::cell::Cell;
use std::ops::Deref;

use chelis_types::manifest::{ManifestedProgram, RootManifest};

use chelis_dag_core::dag::{Dag, DagNode, NodeId, SymbolicDimBinding};
use chelis_ir_core::host::{
    ConcreteHostBinding, ConcreteHostExpr, ConcreteHostFunction, ConcreteHostParam,
    ConcreteHostProgram, HostFunctionOrigin, HostFunctionSpecialization, HostTensorHelper,
    HostTensorInput, HostTensorSpecialization, SummaryRejection,
};

mod storage;

use chelis_dag_core::axis_sources::{LocalGuardClaim, LocalGuardSite};
pub use storage::{
    CStorageLane, HipStorageLane, ReusableOwnedStorage, StorageLane, VerifiedStorageLayout,
    VerifiedStoragePlan, plan_c_storage, plan_c_storage_layout, plan_hip_storage,
};

/// Immutable cursor over the exact verified DAG payload.
/// The raw [`Dag`] remains private so a backend can inspect only the payload whose ownership plan was verified.
/// Without recovering an unchecked sibling graph.
#[derive(Clone, Copy)]
pub struct VerifiedDagView<'a> {
    dag: &'a Dag,
    plan: &'a DagOwnershipPlan,
}

impl<'a> VerifiedDagView<'a> {
    pub(crate) fn analysis(self) -> DagAnalysisView<'a> {
        DagAnalysisView::new(self.dag, self.plan.capacity_scope)
    }
    /// The declaration `decl` names in the verified graph.
    pub fn declaration(
        self,
        decl: chelis_dag_core::dag::DeclId,
    ) -> &'a chelis_dag_core::dag::Declaration {
        self.dag.declaration(decl)
    }

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

    /// The trap seed and activation-gate queries over this graph ([`chelis_dag_core::dag::TrapSeeds`].
    /// Whether a node checks nothing where its activation is false, the one gate declaration, and the literal result claims a witness checks).
    /// An emitter takes one per graph.
    pub fn trap_seeds(self) -> chelis_dag_core::dag::TrapSeeds<'a> {
        self.dag.trap_seeds()
    }

    pub fn topological_order(self) -> Vec<NodeId> {
        self.dag.topological_order()
    }

    /// Every name this graph renders as a C identifier, paired with the origin that produces its value (chelis#665, C4.4).
    /// A declaration consumer reads this instead of searching for a `Load` whose type carries a matching string.
    pub fn dim_extent_origins(self) -> Vec<(String, chelis_dag_core::axis_sources::ExtentOrigin)> {
        chelis_dag_core::axis_sources::dim_extent_origins(self.dag)
    }

    /// The names this graph renders that resolve to no origin.
    /// A lane turns each into a typed receipt.
    /// the legacy walk panicked instead.
    pub fn unresolved_dim_names(self) -> Vec<String> {
        chelis_dag_core::axis_sources::unresolved_dim_names(self.dag)
    }

    /// Every name this graph can render as an identifier, whether or not the entry supplies its extent.
    /// An emitter checks its own declarations against this set.
    pub fn rendered_dim_names(self) -> Vec<String> {
        chelis_dag_core::axis_sources::rendered_dim_names(self.dag)
    }

    /// The INTERFACE bindings: the declaration and entry-guard set.
    /// Derived from the class witnesses rather than recovered by walking for a `Load` that carries a matching string (chelis#1277).
    pub fn symbolic_bindings_interface(self) -> Vec<SymbolicDimBinding> {
        chelis_dag_core::dag::symbolic_bindings_interface(self.dag)
    }

    /// The runtime-dimension equality classes whose guards `spec/04` section 4.7 places at function ENTRY.
    /// Because every operand they compare is an interface value.
    /// The view answers the placement question rather than handing out the graph.
    /// Placement has to resolve a slot's producer to decide whether an operand is an input tensor's.
    /// Axis, a scalar parameter, or a computed value.
    ///
    /// And an emitter doing that itself would reach behind this façade to re-derive what the view already knows.
    /// C1.3's local guard sites: `(node id, axis)` paired with the claim each site guards against.
    /// A `Local` class's guard "takes the source position of the operation that introduces the guarded extent".
    /// (`spec/04-type-system.md` section 4.7), so unlike the entry classes these are keyed by node.
    /// Only `InputAxis` members are sites: an `ExternalAxis` member is the prologue's canonical declaration.
    ///
    /// And an `OpComputed` member's guard needs the derivation narrowed first.
    /// Because the claim over a statically determined operation output (a matmul's `Literal(64)` axis) would guard a value against itself.
    /// The `Load` whose axis a class member's operand names, and that axis.
    /// The view is where the C and HIP lanes ask every class question under chelis#1538's discipline.
    /// So this one is asked here too rather than by reaching around the facade for a raw `Dag`.
    ///
    /// see [`(node id, axis)`0] for what it answers and why three consumers share it.
    pub fn member_load_axis(
        self,
        member: &chelis_dag_core::axis_sources::ClassMember,
    ) -> Option<(NodeId, usize)> {
        chelis_dag_core::axis_sources::member_load_axis(self.dag, member)
    }

    /// Closed primitive form from the verified IR rank relation.
    /// Backends and local extent diagnostics consume the same derivation.
    pub fn expansion_kind(self, id: NodeId) -> chelis_dag_core::axis_sources::ExpansionKind {
        chelis_dag_core::axis_sources::expansion_kind(self.dag, id)
            .expect("verified expansion node has a rank-preserving or inserting form")
    }

    /// C1.3's local guard sites for this verified payload.
    /// The derivation is [`chelis_dag_core::axis_sources::local_dim_guard_sites`], which the DAG evaluator also reads.
    /// So the two lanes cannot place a local guard at two different sites or compare it against two different values.
    /// The view asks it here rather than handing a backend a raw [`Dag`], under the same chelis#1538 discipline as [`Self::member_load_axis`].
    pub fn local_dim_guard_sites(self) -> Result<Vec<(LocalGuardSite, LocalGuardClaim)>, String> {
        chelis_dag_core::axis_sources::local_dim_guard_sites(self.dag)
    }

    pub fn result_extent_sites(
        self,
        root: NodeId,
    ) -> Vec<chelis_dag_core::axis_sources::ResultExtentSite> {
        chelis_dag_core::axis_sources::result_extent_sites(self.dag, root)
    }

    /// The complete positive-rank operand relation for a verified same-shape result.
    /// Verification rejects malformed or empty relations before a backend can get this view.
    pub fn same_shape_result_agreement(
        self,
        node: NodeId,
    ) -> Option<chelis_dag_core::axis_sources::SameShapeAgreement> {
        chelis_dag_core::axis_sources::same_shape_result_agreement(self.dag, node)
            .expect("verified same-shape result agreement")
    }

    pub fn entry_dim_classes(self) -> Vec<chelis_dag_core::axis_sources::RuntimeDimClass> {
        chelis_dag_core::axis_sources::derive_runtime_dim_classes(self.dag)
            .into_iter()
            .filter(|class| {
                class.placement(self.dag) == chelis_dag_core::axis_sources::GuardPlacement::Entry
            })
            .collect()
    }

    /// Individual interface checks, shared with Eval and already scheduled.
    pub fn entry_extent_guards(self) -> Vec<chelis_dag_core::axis_sources::EntryExtentGuard> {
        chelis_dag_core::axis_sources::entry_extent_guards(self.dag)
    }

    /// The same ordered input admission plan consumed by DAG evaluation.
    pub fn entry_validation_plan(self) -> Vec<chelis_dag_core::axis_sources::EntryValidationStep> {
        chelis_dag_core::axis_sources::entry_validation_plan(self.dag)
    }

    /// The named witness claims the entry schedule above already compares.
    /// So an emitter checks each such pair once (`spec/04-type-system.md` §4.7).
    /// The view answers this for the same reason it answers placement: the question needs the class derivation, not the graph.
    pub fn entry_covered_witness_claims(self) -> Vec<(NodeId, usize)> {
        chelis_dag_core::axis_sources::entry_covered_witness_claims(self.dag)
    }

    /// Is this witness retained purely as a section 4.7 entry obligation, with nothing reading its value?
    /// A target whose device lane excludes runtime shape reads asks this to tell an obligation it can discharge.
    /// In its host prologue from a read it must refuse.
    pub fn witness_is_entry_obligation(self, id: NodeId) -> bool {
        chelis_dag_core::axis_sources::witness_is_entry_obligation(self.dag, id)
    }

    /// The obligations that witness owes, reduced to input reads, or `None` when one of them cannot be rendered from the interface alone.
    pub fn witness_entry_obligations(
        self,
        id: NodeId,
    ) -> Option<Vec<chelis_dag_core::axis_sources::WitnessEntryObligation>> {
        chelis_dag_core::axis_sources::witness_entry_obligations(self.dag, id)
    }

    /// The unit-extent claims whose guard section 4.7 places at entry.
    /// The sibling of [`Self::entry_dim_classes`], filtered by the same placement rule through the same shared predicate.
    pub fn entry_unit_extent_claims(self) -> Vec<chelis_dag_core::axis_sources::UnitExtentClaim> {
        chelis_dag_core::axis_sources::derive_unit_extent_claims(self.dag)
            .into_iter()
            .filter(|claim| {
                claim.placement(self.dag) == chelis_dag_core::axis_sources::GuardPlacement::Entry
            })
            .collect()
    }

    /// The `(Load, axis)` each entry-placed unit-extent claim reads.
    /// Both compiled lanes need the same three steps: derive the claims.
    /// Keep the ones section 4.7 places at entry, and resolve each to the input tensor axis whose extent the guard compares.
    /// Doing it here rather than twice is the same discipline `member_load_axis` records.
    /// The C emitter's `member_input_slot` deliberately admits only the folded-read spelling.
    ///
    /// Because a `Load`'s own axis is already declared by the binding loop.
    /// And reusing it here would silently drop every claim whose operand IS an input tensor, which is the common case.
    pub fn entry_unit_extent_reads(self) -> Vec<(NodeId, usize)> {
        self.entry_unit_extent_claims()
            .iter()
            .filter_map(|claim| {
                chelis_dag_core::axis_sources::member_load_axis(self.dag, &claim.member())
            })
            .collect()
    }

    pub fn symbolic_params(self) -> Vec<String> {
        chelis_dag_core::dag::symbolic_params(self.dag)
    }

    pub fn reduction_inlined_fused_elems(self) -> chelis_unord::UnordSet<NodeId> {
        chelis_dag_core::fuse::reduction_inlined_fused_elems(self.dag)
    }

    pub fn reduction_inlined_products(self) -> Vec<chelis_dag_core::fuse::InlinedProduct> {
        chelis_dag_core::fuse::reduction_inlined_products(self.dag)
    }

    /// More checked recipes, leaving the existing specialized product kernel's component nodes to that kernel's independent metadata path.
    pub fn contraction_views(self) -> chelis_ir_core::contraction_views::ContractionViews {
        let products = self.reduction_inlined_products();
        let specialized = products
            .iter()
            .flat_map(chelis_dag_core::fuse::InlinedProduct::skipped)
            .collect::<Vec<_>>();
        chelis_ir_core::contraction_views::contraction_views_with_products(
            self.dag,
            &[],
            &specialized,
            &products,
        )
    }

    pub fn first_integer_abs_node(self) -> Option<NodeId> {
        chelis_dag_core::analysis::first_integer_abs_node(self.dag)
    }

    pub fn first_fused_integer_abs_node(self) -> Option<NodeId> {
        chelis_dag_core::analysis::first_fused_integer_abs_node(self.dag)
    }

    pub fn check_axis_sources(
        self,
        stage: chelis_checker_core::unsupported::Stage,
    ) -> Result<(), chelis_checker_core::unsupported::Unsupported> {
        chelis_ir_core::axis_admission::check_axis_sources(self.dag, stage)
    }

    /// Every name a lane renders as an identifier resolves to one origin (chelis#665).
    /// An emission boundary calls this.
    /// lowering, capacity planning and evaluation do not, for the reason the derivation records.
    pub fn check_rendered_dim_origins(
        self,
        stage: chelis_checker_core::unsupported::Stage,
    ) -> Result<(), chelis_checker_core::unsupported::Unsupported> {
        chelis_ir_core::axis_admission::check_rendered_dim_origins(self.dag, stage)
    }

    /// The exact verified ownership directive attached to `node`.
    /// Every retained DAG node has exactly one directive.
    /// backends use this typed view instead of rediscovering terminal ownership from `RiscOp`.
    pub fn action_for_node(self, node: NodeId) -> Option<VerifiedDagAction<'a>> {
        self.plan
            .directives
            .iter()
            .take(self.dag.len())
            .find(|directive| directive.node() == Some(node))
            .map(verified_dag_action)
    }

    /// All verified DAG actions, including root transfers and scope drops.
    pub fn actions(self) -> impl ExactSizeIterator<Item = VerifiedDagAction<'a>> + 'a {
        self.plan.directives.iter().map(verified_dag_action)
    }
}

/// Closed backend view of a verified DAG ownership directive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifiedDagAction<'a> {
    BorrowLoad { node: NodeId },
    Produce { node: NodeId, borrows: &'a [NodeId] },
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

/// One immutable record in the verified host payload/site bijection.
#[derive(Clone, Copy)]
pub struct VerifiedHostSiteView<'a> {
    record: &'a ir::HostSiteRecord,
    program: &'a ir::OwnershipProgram,
    verification: &'a verify::HostVerification<VerifiedLiveSetBound>,
}

/// Closed classification for a verified ownership directive attached to a host-emission site.
/// The verifier's block, operation, and owner identities remain private.
/// consumers can branch only on this typed semantic role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifiedHostSiteActionKind {
    Operation,
    Terminator,
    ControlEdge,
    ManifestRoot,
}

/// Opaque identity for one verified logical owner.
/// Backends may associate an emitted value with this identity.
/// But cannot manufacture an owner from an integer or infer ownership from a source spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VerifiedOwnerId {
    key: u32,
}

/// Opaque identity for one verified ownership block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VerifiedBlockId {
    key: u32,
}

/// Opaque identity for one verified ownership unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VerifiedUnitId {
    key: u32,
}

/// Feature-only view of a source expression at its ownership-verified host site, before host ABI erasure.
/// This is not a Surf elaboration certificate.
#[cfg(feature = "lowering-trace")]
#[derive(Clone, Copy)]
pub struct VerifiedHostSourceSite<'a> {
    site: VerifiedHostSiteView<'a>,
    expression: &'a ConcreteHostExpr,
}

#[cfg(feature = "lowering-trace")]
impl<'a> VerifiedHostSourceSite<'a> {
    pub fn site(self) -> VerifiedHostSiteView<'a> {
        self.site
    }
    pub fn expression(self) -> &'a ConcreteHostExpr {
        self.expression
    }
    /// Observation-only words in distinct unit and host-site namespaces.
    pub fn unit_word(self) -> usize {
        self.site.record.unit
    }
    pub fn site_word(self) -> usize {
        self.site.id().index()
    }
    pub fn direct_callee_word(self) -> Option<usize> {
        self.direct_callee().map(|callee| callee.key as usize)
    }
    pub fn direct_callee(self) -> Option<VerifiedUnitId> {
        let mut callees = self.site.actions().filter_map(|action| match action {
            VerifiedHostAction::Operation(VerifiedHostOperation::Apply {
                kind: VerifiedApplyKind::DirectCall { callee, .. },
                ..
            }) => Some(callee),
            _ => None,
        });
        let callee = callees.next()?;
        callees.next().is_none().then_some(callee)
    }
}

/// Opaque stable identity for one operation within a verified unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VerifiedOperationId {
    key: u32,
}

/// Closed operand disposition exported by the verified boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifiedOwnershipUse {
    Borrow,
    Move,
    Clone,
}

/// Closed application class exported by the verified boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifiedApplyKind {
    Intrinsic,
    IndirectCall,
    NativeProviderCall,
    /// The indirect callee owner was produced by this closed key operation.
    CheckedBuiltinCall(chelis_ir_core::host_type_state::CheckedBuiltinCallable),
    DirectCall {
        callee: VerifiedUnitId,
        tail: bool,
    },
}

/// Closed edge-local terminal domain.
/// No definition, application, or clone can be smuggled into a selected control-flow edge.
#[derive(Debug, Clone, Copy)]
pub enum VerifiedTerminalView<'a> {
    Drop {
        operation: VerifiedOperationId,
        owner: VerifiedOwnerView<'a>,
    },
    Discard {
        operation: VerifiedOperationId,
        owner: VerifiedOwnerView<'a>,
    },
}

impl VerifiedTerminalView<'_> {
    pub fn operation(self) -> VerifiedOperationId {
        match self {
            Self::Drop { operation, .. } | Self::Discard { operation, .. } => operation,
        }
    }
}

/// Read-only metadata for one verified logical owner.
#[derive(Debug, Clone, Copy)]
pub struct VerifiedOwnerView<'a> {
    id: VerifiedOwnerId,
    info: &'a ir::OwnerInfo,
}

impl<'a> VerifiedOwnerView<'a> {
    pub fn id(self) -> VerifiedOwnerId {
        self.id
    }

    pub fn ty(self) -> &'a chelis_ir_core::host_type_state::ConcreteHostType {
        &self.info.ty
    }

    pub fn is_heap(self) -> bool {
        self.info.class.is_heap()
    }
}

/// A binder spelling projected by one exact, verified binding operation.
/// This capability is intentionally separate from [`VerifiedOwnerView`]: a clone, drop, return.
/// Or edge operand cannot expose source spelling to a backend and therefore cannot seed backend-local lifetime reconstruction.
#[derive(Debug, Clone, Copy)]
pub struct VerifiedBindingName<'a> {
    name: &'a str,
}

impl<'a> VerifiedBindingName<'a> {
    pub fn as_str(self) -> &'a str {
        self.name
    }
}

/// Exact association between one verified function-body owner and the retained payload spelling that binds it.
/// This includes declared parameters and verifier-certified captured globals.
#[derive(Debug, Clone, Copy)]
pub struct VerifiedHostBodyBinding<'a> {
    owner: VerifiedOwnerView<'a>,
    name: &'a str,
}

impl<'a> VerifiedHostBodyBinding<'a> {
    pub fn owner(self) -> VerifiedOwnerView<'a> {
        self.owner
    }

    pub fn name(self) -> &'a str {
        self.name
    }
}

/// One typed use of a verified logical owner.
#[derive(Debug, Clone, Copy)]
pub struct VerifiedOperandView<'a> {
    owner: VerifiedOwnerView<'a>,
    use_: VerifiedOwnershipUse,
}

/// One verified control-flow edge, including the exact owned/borrowed values supplied to the target block parameters.
#[derive(Debug, Clone)]
pub struct VerifiedEdgeView<'a> {
    target: VerifiedBlockId,
    params: Vec<VerifiedOwnerView<'a>>,
    args: Vec<VerifiedOperandView<'a>>,
    terminals: Vec<VerifiedTerminalView<'a>>,
}

impl<'a> VerifiedEdgeView<'a> {
    pub fn target(&self) -> VerifiedBlockId {
        self.target
    }

    pub fn params(&self) -> &[VerifiedOwnerView<'a>] {
        &self.params
    }

    pub fn args(&self) -> &[VerifiedOperandView<'a>] {
        &self.args
    }

    pub fn terminals(&self) -> &[VerifiedTerminalView<'a>] {
        &self.terminals
    }
}

impl<'a> VerifiedOperandView<'a> {
    pub fn owner(self) -> VerifiedOwnerView<'a> {
        self.owner
    }

    pub fn use_(self) -> VerifiedOwnershipUse {
        self.use_
    }
}

/// Closed, typed operation directive.
/// Labels are diagnostic renderings only.
/// ownership behavior is selected exclusively by these variants and operand dispositions.
#[derive(Debug, Clone)]
pub enum VerifiedHostOperation<'a> {
    Define {
        operation: VerifiedOperationId,
        block: VerifiedBlockId,
        dest: VerifiedOwnerView<'a>,
        label: &'a str,
    },
    Apply {
        operation: VerifiedOperationId,
        block: VerifiedBlockId,
        dest: Option<VerifiedOwnerView<'a>>,
        binding_name: Option<VerifiedBindingName<'a>>,
        label: &'a str,
        kind: VerifiedApplyKind,
        args: Vec<VerifiedOperandView<'a>>,
    },
    Clone {
        operation: VerifiedOperationId,
        block: VerifiedBlockId,
        dest: VerifiedOwnerView<'a>,
        source: VerifiedOperandView<'a>,
    },
    /// A non-owning projection of a logical owner into the host payload slot at this exact site.
    Project {
        operation: VerifiedOperationId,
        block: VerifiedBlockId,
        source: VerifiedOperandView<'a>,
    },
    LoopItem {
        operation: VerifiedOperationId,
        block: VerifiedBlockId,
        dest: VerifiedOwnerView<'a>,
        list: VerifiedOperandView<'a>,
    },
    Drop {
        operation: VerifiedOperationId,
        block: VerifiedBlockId,
        owner: VerifiedOperandView<'a>,
    },
    Discard {
        operation: VerifiedOperationId,
        block: VerifiedBlockId,
        owner: VerifiedOwnerView<'a>,
    },
    RootConsume {
        operation: VerifiedOperationId,
        block: VerifiedBlockId,
        root: &'a str,
        owner: VerifiedOperandView<'a>,
    },
}

/// Closed, typed terminal directive for one verified ownership block.
#[derive(Debug, Clone)]
pub enum VerifiedHostTerminator<'a> {
    Return {
        block: VerifiedBlockId,
        result: VerifiedOperandView<'a>,
    },
    Jump {
        block: VerifiedBlockId,
        edge: VerifiedEdgeView<'a>,
    },
    Branch {
        block: VerifiedBlockId,
        condition: VerifiedOperandView<'a>,
        then_edge: VerifiedEdgeView<'a>,
        else_edge: VerifiedEdgeView<'a>,
    },
    Match {
        block: VerifiedBlockId,
        scrutinee: VerifiedOperandView<'a>,
        arms: Vec<VerifiedEdgeView<'a>>,
    },
    Loop {
        block: VerifiedBlockId,
        list: VerifiedOperandView<'a>,
        body_edge: VerifiedEdgeView<'a>,
        exit_edge: VerifiedEdgeView<'a>,
    },
    Exit {
        block: VerifiedBlockId,
    },
}

/// One exact semantic directive attached to a verified host-emission site.
#[derive(Debug, Clone)]
pub enum VerifiedHostAction<'a> {
    Operation(VerifiedHostOperation<'a>),
    Terminator(VerifiedHostTerminator<'a>),
    ControlEdge {
        source: VerifiedBlockId,
        edge: VerifiedEdgeView<'a>,
    },
    ManifestRoot {
        manifest_index: Option<usize>,
        owner: VerifiedOwnerView<'a>,
    },
}

impl<'a> VerifiedHostSiteView<'a> {
    pub fn id(self) -> HostSiteId {
        self.record.id
    }

    pub fn kind(self) -> HostSiteKind {
        self.record.kind
    }

    pub fn action_kinds(self) -> impl ExactSizeIterator<Item = VerifiedHostSiteActionKind> + 'a {
        self.record.actions.iter().map(|action| match action {
            ir::HostSiteAction::Operation { .. } => VerifiedHostSiteActionKind::Operation,
            ir::HostSiteAction::Terminator { .. } => VerifiedHostSiteActionKind::Terminator,
            ir::HostSiteAction::ControlEdge { .. } => VerifiedHostSiteActionKind::ControlEdge,
            ir::HostSiteAction::Root { .. } => VerifiedHostSiteActionKind::ManifestRoot,
        })
    }

    pub fn actions(self) -> impl ExactSizeIterator<Item = VerifiedHostAction<'a>> + 'a {
        self.record.actions.iter().map(move |action| {
            verified_host_action(self.program, self.verification, action)
                .expect("verified host site action references checked ownership IR")
        })
    }
}

fn verified_owner<'a>(
    program: &'a ir::OwnershipProgram,
    unit: usize,
    owner: ir::OwnerId,
) -> Option<VerifiedOwnerView<'a>> {
    let info = program.units.get(unit)?.owners.get(&owner)?;
    Some(VerifiedOwnerView {
        id: VerifiedOwnerId { key: owner.0 },
        info,
    })
}

fn verified_operand<'a>(
    program: &'a ir::OwnershipProgram,
    unit: usize,
    operand: &ir::Operand,
) -> Option<VerifiedOperandView<'a>> {
    let use_ = match operand.use_ {
        ir::OwnershipUse::Borrow => VerifiedOwnershipUse::Borrow,
        ir::OwnershipUse::Move => VerifiedOwnershipUse::Move,
        ir::OwnershipUse::Clone => VerifiedOwnershipUse::Clone,
    };
    Some(VerifiedOperandView {
        owner: verified_owner(program, unit, operand.owner)?,
        use_,
    })
}

fn verified_edge<'a>(
    program: &'a ir::OwnershipProgram,
    unit: usize,
    edge: &ir::Edge,
) -> Option<VerifiedEdgeView<'a>> {
    let block = program
        .units
        .get(unit)?
        .blocks
        .get(edge.target.0 as usize)?;
    Some(VerifiedEdgeView {
        target: VerifiedBlockId { key: edge.target.0 },
        params: block
            .params
            .iter()
            .map(|param| verified_owner(program, unit, param.owner))
            .collect::<Option<Vec<_>>>()?,
        args: edge
            .args
            .iter()
            .map(|operand| verified_operand(program, unit, operand))
            .collect::<Option<Vec<_>>>()?,
        terminals: edge
            .terminals
            .iter()
            .map(|terminal| match terminal.kind {
                ir::Terminal::Drop(owner) => {
                    verified_owner(program, unit, owner).map(|owner| VerifiedTerminalView::Drop {
                        operation: VerifiedOperationId { key: terminal.id.0 },
                        owner,
                    })
                }
                ir::Terminal::Discard(owner) => verified_owner(program, unit, owner).map(|owner| {
                    VerifiedTerminalView::Discard {
                        operation: VerifiedOperationId { key: terminal.id.0 },
                        owner,
                    }
                }),
            })
            .collect::<Option<Vec<_>>>()?,
    })
}

fn verified_host_action<'a>(
    program: &'a ir::OwnershipProgram,
    verification: &'a verify::HostVerification<VerifiedLiveSetBound>,
    action: &'a ir::HostSiteAction,
) -> Option<VerifiedHostAction<'a>> {
    let block_id = |block: ir::BlockId| VerifiedBlockId { key: block.0 };
    match *action {
        ir::HostSiteAction::Operation {
            unit,
            block,
            operation,
        } => {
            let locations = verification.locations().observed(Some(&SiteObserver));
            locations.block(program, unit, block)?;
            if let Some((_, terminal)) = locations.terminal(program, unit, block, operation) {
                let operation = VerifiedOperationId { key: operation.0 };
                let block = block_id(block);
                return Some(VerifiedHostAction::Operation(match terminal.kind {
                    ir::Terminal::Drop(owner) => VerifiedHostOperation::Drop {
                        operation,
                        block,
                        owner: verified_operand(program, unit, &ir::Operand::move_(owner))?,
                    },
                    ir::Terminal::Discard(owner) => VerifiedHostOperation::Discard {
                        operation,
                        block,
                        owner: verified_owner(program, unit, owner)?,
                    },
                }));
            }
            let op = locations.operation(program, unit, block, operation)?;
            let block = block_id(block);
            let operation_id = VerifiedOperationId { key: op.id.0 };
            Some(VerifiedHostAction::Operation(match &op.kind {
                ir::Op::Define { dest, label } => VerifiedHostOperation::Define {
                    operation: operation_id,
                    block,
                    dest: verified_owner(program, unit, *dest)?,
                    label,
                },
                ir::Op::Apply {
                    dest,
                    label,
                    kind,
                    args,
                    ..
                } => {
                    let binding_name = dest
                        .and_then(|owner| program.units.get(unit)?.owners.get(&owner))
                        .and_then(|info| info.names.first())
                        .map(|name| VerifiedBindingName { name });
                    VerifiedHostOperation::Apply {
                        operation: operation_id,
                        block,
                        dest: dest.and_then(|owner| verified_owner(program, unit, owner)),
                        binding_name,
                        label,
                        kind: match kind {
                            ir::ApplyKind::Intrinsic => VerifiedApplyKind::Intrinsic,
                            ir::ApplyKind::IndirectCall => VerifiedApplyKind::IndirectCall,
                            ir::ApplyKind::NativeProviderCall => {
                                VerifiedApplyKind::NativeProviderCall
                            }
                            ir::ApplyKind::CheckedBuiltinCall(op) => {
                                VerifiedApplyKind::CheckedBuiltinCall(*op)
                            }
                            ir::ApplyKind::DirectCall { callee } => VerifiedApplyKind::DirectCall {
                                callee: VerifiedUnitId { key: callee.0 },
                                tail: verification.is_tail_call(program.units.get(unit)?.id, op.id),
                            },
                        },
                        args: args
                            .iter()
                            .map(|operand| verified_operand(program, unit, operand))
                            .collect::<Option<Vec<_>>>()?,
                    }
                }
                ir::Op::Copy { dest, source } => VerifiedHostOperation::Clone {
                    operation: operation_id,
                    block,
                    dest: verified_owner(program, unit, *dest)?,
                    source: verified_operand(program, unit, source)?,
                },
                ir::Op::Project { source } => VerifiedHostOperation::Project {
                    operation: operation_id,
                    block,
                    source: verified_operand(program, unit, source)?,
                },
                ir::Op::LoopItem { dest, list } => VerifiedHostOperation::LoopItem {
                    operation: operation_id,
                    block,
                    dest: verified_owner(program, unit, *dest)?,
                    list: verified_operand(program, unit, list)?,
                },
                ir::Op::Drop { owner } => VerifiedHostOperation::Drop {
                    operation: operation_id,
                    block,
                    owner: verified_operand(program, unit, owner)?,
                },
                ir::Op::Discard { owner } => VerifiedHostOperation::Discard {
                    operation: operation_id,
                    block,
                    owner: verified_owner(program, unit, *owner)?,
                },
                ir::Op::RootConsume { root, owner } => VerifiedHostOperation::RootConsume {
                    operation: operation_id,
                    block,
                    root,
                    owner: verified_operand(program, unit, owner)?,
                },
            }))
        }
        ir::HostSiteAction::Terminator { unit, block } => {
            let terminal = &program
                .units
                .get(unit)?
                .blocks
                .get(block.0 as usize)?
                .terminator;
            let block = block_id(block);
            Some(VerifiedHostAction::Terminator(match terminal {
                ir::Terminator::Return { result } => VerifiedHostTerminator::Return {
                    block,
                    result: verified_operand(program, unit, result)?,
                },
                ir::Terminator::Jump(edge) => VerifiedHostTerminator::Jump {
                    block,
                    edge: verified_edge(program, unit, edge)?,
                },
                ir::Terminator::Branch {
                    condition,
                    then_edge,
                    else_edge,
                } => VerifiedHostTerminator::Branch {
                    block,
                    condition: verified_operand(program, unit, condition)?,
                    then_edge: verified_edge(program, unit, then_edge)?,
                    else_edge: verified_edge(program, unit, else_edge)?,
                },
                ir::Terminator::Match { scrutinee, arms } => VerifiedHostTerminator::Match {
                    block,
                    scrutinee: verified_operand(program, unit, scrutinee)?,
                    arms: arms
                        .iter()
                        .map(|edge| verified_edge(program, unit, edge))
                        .collect::<Option<Vec<_>>>()?,
                },
                ir::Terminator::Loop {
                    list,
                    body_edge,
                    exit_edge,
                } => VerifiedHostTerminator::Loop {
                    block,
                    list: verified_operand(program, unit, list)?,
                    body_edge: verified_edge(program, unit, body_edge)?,
                    exit_edge: verified_edge(program, unit, exit_edge)?,
                },
                ir::Terminator::Exit => VerifiedHostTerminator::Exit { block },
            }))
        }
        ir::HostSiteAction::ControlEdge {
            unit,
            edge: edge_id,
            source,
            target,
        } => {
            let edge = verification
                .locations()
                .observed(Some(&SiteObserver))
                .edge(program, unit, source, edge_id, target)?;
            Some(VerifiedHostAction::ControlEdge {
                source: block_id(source),
                edge: verified_edge(program, unit, edge)?,
            })
        }
        ir::HostSiteAction::Root {
            unit,
            manifest_index,
            owner,
        } => Some(VerifiedHostAction::ManifestRoot {
            manifest_index,
            owner: verified_owner(program, unit, owner)?,
        }),
    }
}

/// Immutable cursor over one nested host tensor helper and its verified DAG.
#[derive(Clone, Copy)]
pub struct VerifiedHostTensorHelperView<'a> {
    helper: &'a HostTensorHelper,
    dag: VerifiedDagView<'a>,
    #[cfg(feature = "lowering-trace")]
    source_location: (usize, usize),
}

impl<'a> VerifiedHostTensorHelperView<'a> {
    /// Observation-only owning unit and helper-slot words.
    /// not function labels.
    #[cfg(feature = "lowering-trace")]
    pub fn source_location(self) -> (usize, usize) {
        self.source_location
    }
    pub fn name(self) -> &'a str {
        &self.helper.name
    }

    pub fn inputs(self) -> &'a [HostTensorInput] {
        &self.helper.inputs
    }

    pub fn output(self) -> &'a chelis_dag_core::dag::TensorType {
        &self.helper.output
    }

    pub fn specialization(self) -> Option<&'a HostTensorSpecialization> {
        self.helper.specialization.as_ref()
    }

    /// The input this helper returns unchanged, by the one definition the ownership lowering read ([`chelis_ir_core::host::HostTensorHelper::identity_input`]).
    pub fn identity_input(self) -> Option<&'a HostTensorInput> {
        self.helper.identity_input()
    }

    pub fn summary_rejection(self) -> Option<&'a chelis_ir_core::host::HelperSummaryRejection> {
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

    pub fn helper_result_claim_axes(self) -> &'a [chelis_dag_core::dag::RtAxis] {
        &self.function().helper_result_claim_axes
    }

    pub fn result_claim(
        self,
    ) -> Option<&'a std::sync::Arc<chelis_ir_core::claim_pattern::ClaimPattern>> {
        self.function().result_claim.as_ref()
    }

    pub fn entry_claims(
        self,
    ) -> &'a [Option<std::sync::Arc<chelis_ir_core::claim_pattern::ClaimPattern>>] {
        &self.function().entry_claims
    }

    pub fn entry_contract(
        self,
    ) -> &'a chelis_ir_core::host::EntryContract<chelis_ir_core::host_type_state::ConcreteHostType>
    {
        &self.function().entry_contract
    }

    pub fn params(self) -> &'a [ConcreteHostParam] {
        &self.function().params
    }

    /// Function-body owner identities paired with their exact payload parameters or captured globals.
    /// Authored entry-adapter owners are deliberately excluded: their verified jump transfers into these consuming body owners.
    pub fn body_bindings(self) -> Vec<VerifiedHostBodyBinding<'a>> {
        let unit_index = self.index + 1;
        let unit = &self.emission.ownership_program().units[unit_index];
        let entry = unit
            .blocks
            .iter()
            .find(|block| block.id == unit.entry)
            .expect("verified function entry block");
        let body_id = if self.origin() == HostFunctionOrigin::Authored {
            match &entry.terminator {
                ir::Terminator::Jump(edge) => edge.target,
                _ => unreachable!("verified authored function entry adapter"),
            }
        } else {
            unit.entry
        };
        let body = unit
            .blocks
            .iter()
            .find(|block| block.id == body_id)
            .expect("verified function body block");
        assert!(
            body.params.len() >= self.function().params.len(),
            "verified function body omits declared parameters"
        );
        body.params
            .iter()
            .enumerate()
            .map(|(index, param)| {
                let owner =
                    verified_owner(self.emission.ownership_program(), unit_index, param.owner)
                        .expect("verified function body owner");
                let name = if let Some(payload) = self.function().params.get(index) {
                    payload.name.as_str()
                } else {
                    // The capture is identified by the spelling it was minted under, never by its owner's alias names.
                    // A later local alias may be spelled like any top-level value.
                    let name = unit
                        .captured_globals
                        .get(&param.owner)
                        .expect("verified environment parameter records its captured global");
                    let source = chelis_dag_core::LoadStoreName::top_level_source_for_label(name)
                        .ok()
                        .flatten()
                        .unwrap_or_else(|| name.clone());
                    assert!(
                        self.emission
                            .payload
                            .program
                            .globals
                            .iter()
                            .any(|binding| binding.name == source),
                        "verified capture `{name}` names a retained global"
                    );
                    name.as_str()
                };
                VerifiedHostBodyBinding { owner, name }
            })
            .collect()
    }

    pub fn ret_ty(self) -> &'a chelis_ir_core::host_type_state::ConcreteHostType {
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
        let plan = self.emission.nested_dags.iter().find(|candidate| {
            candidate.location
                == NestedDagLocation::Function {
                    function: self.index,
                    helper,
                }
        })?;
        Some(VerifiedHostTensorHelperView {
            helper: raw,
            #[cfg(feature = "lowering-trace")]
            source_location: (self.index + 1, helper),
            dag: VerifiedDagView {
                dag: &raw.dag,
                plan: &plan.plan,
            },
        })
    }

    pub fn sites(self) -> impl Iterator<Item = VerifiedHostSiteView<'a>> + 'a {
        let unit = self.index + 1;
        let program = self.emission.ownership_program();
        let records = self.emission.sites;
        self.emission
            .sites_by_unit
            .of_observed(unit, Some(&SiteObserver))
            .iter()
            .map(move |&position| VerifiedHostSiteView {
                record: records
                    .record_observed(position, Some(&SiteObserver))
                    .expect("indexed site record"),
                program,
                verification: self.emission.verification,
            })
    }
}

/// Immutable view of the exact verified host emission payload.
/// Its fields are private.
/// nested DAGs are reachable only as verified child cursors.
#[derive(Clone, Copy)]
pub struct VerifiedHostEmission<'a> {
    payload: &'a HostEmissionPayload,
    program: &'a ir::OwnershipProgram,
    verification: &'a verify::HostVerification<VerifiedLiveSetBound>,
    sites: &'a ir::HostSiteMap,
    sites_by_unit: &'a locations::SitesByUnit,
    nested_dags: &'a [NestedDagProof],
}

impl<'a> VerifiedHostEmission<'a> {
    #[cfg(feature = "lowering-trace")]
    pub fn source_expressions(self) -> Vec<VerifiedHostSourceSite<'a>> {
        source_expressions(self)
    }
    fn ownership_program(self) -> &'a ir::OwnershipProgram {
        // The emission cursor is built only from the host proof variant.
        // Its program reference is threaded explicitly below rather than recovered from a raw sibling payload.
        self.program
    }

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
        let plan = self
            .nested_dags
            .iter()
            .find(|candidate| candidate.location == NestedDagLocation::Global(helper))?;
        Some(VerifiedHostTensorHelperView {
            helper: raw,
            #[cfg(feature = "lowering-trace")]
            source_location: (0, helper),
            dag: VerifiedDagView {
                dag: &raw.dag,
                plan: &plan.plan,
            },
        })
    }

    pub fn summary_rejections(self) -> &'a [SummaryRejection] {
        &self.payload.program.summary_rejections
    }

    /// The constructor layouts of the ADTs the functions' parameters carry.
    pub fn adt_layouts(
        self,
    ) -> &'a [chelis_ir_core::host::HostAdtLayout<
        chelis_ir_core::host_type_state::ConcreteHostType,
    >] {
        &self.payload.program.adt_layouts
    }

    pub fn manifest(self) -> &'a RootManifest {
        &self.payload.manifest
    }

    pub fn sites(self) -> impl ExactSizeIterator<Item = VerifiedHostSiteView<'a>> + 'a {
        let program = self.ownership_program();
        self.sites
            .records_observed(Some(&SiteObserver))
            .map(move |record| VerifiedHostSiteView {
                record,
                program,
                verification: self.verification,
            })
    }

    pub fn root_sites(self) -> impl Iterator<Item = VerifiedHostSiteView<'a>> + 'a {
        let program = self.ownership_program();
        let records = self.sites;
        self.sites_by_unit
            .of_observed(0, Some(&SiteObserver))
            .iter()
            .map(move |&position| VerifiedHostSiteView {
                record: records
                    .record_observed(position, Some(&SiteObserver))
                    .expect("indexed site record"),
                program,
                verification: self.verification,
            })
    }
}
/// Closed payload identity.
/// Public only so the sealed generic verifier can carry a public bound without exposing its private representation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadKind {
    Host,
    Dag,
}

#[derive(Debug)]
pub struct VerifiedLiveSetBound {
    data: chelis_ir_core::ownership::LiveSetBound,
}

impl VerifiedLiveSetBound {
    pub fn max_live_heap_owners(&self) -> usize {
        self.data.max_live_heap_owners
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A sealed emission-payload specialization.
/// Downstream crates can name the concrete host and DAG payloads but cannot add a third unchecked lane.
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

impl Drop for HostEmissionPayload {
    fn drop(&mut self) {
        chelis_ir_core::host::drain_host_program_expressions(&mut self.program);
    }
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
        /// The verification `last_use::schedule` computed for exactly `program`.
        /// No code mutates a constructed proof, so the boundary reuses it instead of verifying the same program again (chelis#3730).
        verification: verify::HostVerification<VerifiedLiveSetBound>,
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

/// Unverified ownership-lowered form.
/// Its fields and constructor are private.
#[derive(Debug)]
pub struct OwnershipProgram<P: EmissionPayload> {
    payload: P,
    proof: OwnershipProof,
}

/// Verified host or DAG payload.
/// Only [`verify_ownership`] constructs it.
#[derive(Debug)]
pub struct VerifiedOwnershipProgram<P: EmissionPayload>(OwnershipProgram<P>, VerificationProof);

/// The specialization [`verify_ownership`] checked.
/// A host program's [`verify::HostVerification`] travels in its [`OwnershipProof::Host`].
#[derive(Debug)]
enum VerificationProof {
    /// The verified site map's records, indexed by unit for the emission views.
    Host(locations::SitesByUnit),
    Dag,
}

pub type HostOwnershipProgram = OwnershipProgram<HostEmissionPayload>;
pub type DagOwnershipProgram = OwnershipProgram<DagEmissionPayload>;
pub type VerifiedHostProgram = VerifiedOwnershipProgram<HostEmissionPayload>;
pub type VerifiedDagProgram = VerifiedOwnershipProgram<DagEmissionPayload>;

/// Consume a concrete host program, materialize every manifested host root, and lower the exact resulting payload to ownership IR.
pub fn lower_host_ownership(
    manifested: &ManifestedProgram,
    host: ConcreteHostProgram,
) -> Result<HostOwnershipProgram, OwnershipError> {
    lower_host_ownership_impl(manifested, host)
}

/// Consume the opaque host carrier, whose helper traces the caller snapshots, and lower its exact host payload to ownership IR.
pub fn lower_host_execution_ownership(
    manifested: &ManifestedProgram,
    plan: crate::host::HostExecutionPlan,
) -> Result<HostOwnershipProgram, OwnershipError> {
    lower_host_ownership_impl(manifested, plan.into_program())
}

fn lower_host_ownership_impl(
    manifested: &ManifestedProgram,
    mut host: ConcreteHostProgram,
) -> Result<HostOwnershipProgram, OwnershipError> {
    crate::resources::with_operation(|operation| {
        let root_bindings =
            lower::materialize_manifest_roots(&mut host, &NativeManifest(manifested.manifest()))?;
        let mut sites = ir::HostSiteBuilder::default();
        let mut program = operation.host.with_native_provider_call(
            crate::host::active_native_provider_call(),
            || {
                lower::lower(
                    manifested.checked().facts(),
                    &host,
                    &NativeManifest(manifested.manifest()),
                    &root_bindings,
                    &mut sites,
                    operation,
                )
            },
        )?;
        let mut sites = sites.finish();
        let verification =
            last_use::schedule_observed(&mut program, &mut sites, operation.resources.observer)?
                .into_verification()
                .map_live_set_bound(|data| VerifiedLiveSetBound { data });
        let nested_dags = lower_nested_dags(&host)?;
        Ok(OwnershipProgram {
            payload: HostEmissionPayload {
                program: host,
                manifest: manifested.manifest().clone(),
            },
            proof: OwnershipProof::Host {
                program,
                sites,
                nested_dags,
                verification,
            },
        })
    })
}

/// Consume a standalone DAG and attach an explicit ownership directive for every node, root, and implicit scope-exit drop.
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
    let verification = match (&program.proof, P::KIND) {
        (
            OwnershipProof::Host {
                program: ir_program,
                sites,
                nested_dags,
                verification: _,
            },
            PayloadKind::Host,
        ) => {
            let payload = host_payload(&program)?;
            verify::verify_host_sites_observed(
                &payload.program,
                &NativeManifest(&payload.manifest),
                ir_program,
                sites,
                Some(&SiteObserver),
            )?;
            verify_manifest_sinks(payload, ir_program, sites)?;
            verify_nested_dags(&program, nested_dags)?;
            VerificationProof::Host(locations::SitesByUnit::new_observed(
                sites,
                Some(&SiteObserver),
            ))
        }
        (OwnershipProof::Dag(plan), PayloadKind::Dag) => {
            let dag = dag_payload(&program)?;
            plan.verify(dag)?;
            VerificationProof::Dag
        }
        _ => {
            return Err(OwnershipError::LoweringInvariant {
                unit: "ownership-boundary".to_string(),
                detail: "payload specialization does not match its private proof".to_string(),
            });
        }
    };
    Ok(VerifiedOwnershipProgram(program, verification))
}

fn verify_manifest_sinks(
    payload: &HostEmissionPayload,
    program: &ir::OwnershipProgram,
    sites: &ir::HostSiteMap,
) -> Result<(), OwnershipError> {
    use chelis_types::types::Lane;
    let expected = payload
        .manifest
        .entries
        .iter()
        .filter(|entry| entry.lane == Lane::Host)
        .map(|entry| entry.name.as_str())
        .collect::<Vec<_>>();
    let actual = sites
        .records_observed(Some(&SiteObserver))
        .filter(|site| {
            site.actions.iter().any(|action| {
                matches!(
                    action,
                    ir::HostSiteAction::Root {
                        manifest_index: Some(_),
                        ..
                    }
                )
            })
        })
        .flat_map(|site| &site.actions)
        .filter_map(|action| match *action {
            ir::HostSiteAction::Operation {
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
            program,
            sites,
            nested_dags,
            verification,
        } = &self.0.proof
        else {
            unreachable!("sealed host specialization")
        };
        let payload = (&self.0.payload as &dyn std::any::Any)
            .downcast_ref::<HostEmissionPayload>()
            .expect("sealed host payload specialization");
        let VerificationProof::Host(sites_by_unit) = &self.1 else {
            unreachable!("sealed host verification specialization")
        };
        VerifiedHostEmission {
            payload,
            program,
            verification,
            sites,
            sites_by_unit,
            nested_dags,
        }
    }

    pub fn live_set_bound(&self) -> &VerifiedLiveSetBound {
        let (OwnershipProof::Host { verification, .. }, VerificationProof::Host(_)) =
            (&self.0.proof, &self.1)
        else {
            unreachable!("sealed host verification specialization")
        };
        verification.live_set_bound()
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
        let OwnershipProof::Dag(plan) = &self.0.proof else {
            unreachable!("sealed DAG specialization")
        };
        VerifiedDagView {
            dag: &payload.dag,
            plan,
        }
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

#[derive(Debug)]
struct DagOwnershipPlan {
    data: chelis_ir_core::ownership::dag::DagOwnershipPlan,
    capacity_scope: chelis_ir_core::capacity_key::CapacityScope,
}
impl Deref for DagOwnershipPlan {
    type Target = chelis_ir_core::ownership::dag::DagOwnershipPlan;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}
impl DagOwnershipPlan {
    fn lower(dag: &Dag) -> Result<Self, OwnershipError> {
        let data = chelis_ir_core::ownership::dag::DagOwnershipPlan::lower(dag)?;
        Ok(Self {
            data,
            capacity_scope: crate::capacity_key::fresh_capacity_scope(),
        })
    }
}

fn verified_dag_action(directive: &DagDirective) -> VerifiedDagAction<'_> {
    match directive {
        DagDirective::BorrowLoad { node } => VerifiedDagAction::BorrowLoad { node: *node },
        DagDirective::Produce { node, borrows } => VerifiedDagAction::Produce {
            node: *node,
            borrows,
        },
        DagDirective::Clone { node, source } => VerifiedDagAction::Clone {
            node: *node,
            source: *source,
        },
        DagDirective::CloneProduce { node, source } => VerifiedDagAction::CloneProduce {
            node: *node,
            source: *source,
        },
        DagDirective::MoveProduce { node, source } => VerifiedDagAction::MoveProduce {
            node: *node,
            source: *source,
        },
        DagDirective::CloneStore { node, source } => VerifiedDagAction::CloneStore {
            node: *node,
            source: *source,
        },
        DagDirective::MoveStore { node, source } => VerifiedDagAction::MoveStore {
            node: *node,
            source: *source,
        },
        DagDirective::StoreRoot { node } => VerifiedDagAction::StoreRoot { node: *node },
        DagDirective::BorrowedDrop { node, source } => VerifiedDagAction::BorrowedDrop {
            node: *node,
            source: *source,
        },
        DagDirective::OwnedDrop { node, source } => VerifiedDagAction::OwnedDrop {
            node: *node,
            source: *source,
        },
        DagDirective::Root { source } => VerifiedDagAction::Root { source: *source },
        DagDirective::RootClone { source } => VerifiedDagAction::RootClone { source: *source },
        DagDirective::ScopeDrop { source } => VerifiedDagAction::ScopeDrop { source: *source },
    }
}

struct NativeManifest<'a>(&'a RootManifest);
impl chelis_ir_core::ownership::manifest::ManifestSource for NativeManifest<'_> {
    fn len(&self) -> usize {
        self.0.entries.len()
    }
    fn entry(
        &self,
        index: usize,
    ) -> Option<chelis_ir_core::ownership::manifest::RootEntryView<'_>> {
        self.0
            .entries
            .get(index)
            .map(|entry| chelis_ir_core::ownership::manifest::RootEntryView {
                name: &entry.name,
                def_name: &entry.def_name,
                path: &entry.path,
                host: entry.lane == chelis_types::types::Lane::Host,
            })
    }
}

thread_local! {
    static SITE_LOOKUP_ENTRIES: Cell<Option<u64>> = const { Cell::new(None) };
    #[cfg(test)]
    static OWNERSHIP_VERIFICATIONS: Cell<u64> = const { Cell::new(0) };
}

pub(crate) fn observe_site_lookup_entries(entries: u64) {
    SITE_LOOKUP_ENTRIES.with(|count| {
        if let Some(value) = count.get() {
            count.set(Some(value + entries));
        }
    });
}

pub(crate) fn observe_ownership_verification() {
    #[cfg(test)]
    OWNERSHIP_VERIFICATIONS.with(|count| count.set(count.get() + 1));
}

struct SiteObserver;
impl chelis_ir_core::resources::Observer for SiteObserver {
    fn ownership_reference_checks_enabled(&self) -> bool {
        cfg!(test)
    }
    fn ownership_lookup_entries(&self, entries: u64) {
        observe_ownership_lookup_entries(entries);
    }
    fn ownership_reference_scan_entries(&self, entries: u64) {
        observe_ownership_reference_scan_entries(entries);
    }
    fn site_lookup_entries(&self, entries: u64) {
        observe_site_lookup_entries(entries);
    }
}

/// Run `work`, counting the entries the host-site lookups visit on this thread meanwhile: each table entry built, and each lookup answered.
#[doc(hidden)]
pub fn counting_site_lookup_entries<R>(work: impl FnOnce() -> R) -> (R, u64) {
    struct Restore(Option<u64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SITE_LOOKUP_ENTRIES.with(|count| count.set(self.0));
        }
    }
    let restore = Restore(SITE_LOOKUP_ENTRIES.with(|count| count.replace(Some(0))));
    let result = work();
    let counted = SITE_LOOKUP_ENTRIES
        .with(Cell::get)
        .expect("the count stays armed while the work runs");
    drop(restore);
    (result, counted)
}

#[cfg(test)]
mod tests;

/// Project only the already-verified, retained source-structural census.
/// No emitted spelling or separately supplied source may construct this cursor.
#[cfg(feature = "lowering-trace")]
fn source_expressions(emission: VerifiedHostEmission<'_>) -> Vec<VerifiedHostSourceSite<'_>> {
    let expected = verify::census_host_payload(
        &emission.payload.program,
        &NativeManifest(&emission.payload.manifest),
    )
    .expect("verified host payload census");
    expected
        .into_iter()
        .zip(emission.sites())
        .filter_map(|(expected, site)| {
            expected
                .expression
                .map(|expression| VerifiedHostSourceSite { site, expression })
        })
        .collect()
}

#[cfg(test)]
thread_local! {
    static OWNERSHIP_LOOKUP_ENTRIES: Cell<u64> = const { Cell::new(0) };
    static OWNERSHIP_REFERENCE_SCANS: Cell<bool> = const { Cell::new(false) };
}

pub(crate) fn observe_ownership_lookup_entries(entries: u64) {
    #[cfg(test)]
    OWNERSHIP_LOOKUP_ENTRIES.with(|count| count.set(count.get() + entries));
    #[cfg(not(test))]
    let _ = entries;
}
pub(crate) fn observe_ownership_reference_scan_entries(entries: u64) {
    #[cfg(test)]
    if OWNERSHIP_REFERENCE_SCANS.with(Cell::get) {
        observe_ownership_lookup_entries(entries);
    }
    #[cfg(not(test))]
    let _ = entries;
}
#[cfg(test)]
fn with_reference_scans_counted<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            OWNERSHIP_REFERENCE_SCANS.with(|flag| flag.set(self.0));
        }
    }
    let restore = Restore(OWNERSHIP_REFERENCE_SCANS.with(|flag| flag.replace(true)));
    let result = run();
    drop(restore);
    result
}
