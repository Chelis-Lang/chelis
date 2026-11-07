use super::dag::DagAnalysisView;
use super::{LiveByteBound, OwnershipError};
use crate::capacity_key::{
    CapacityKey, CapacityKeyBuild, capacity_key_for_node, shape_capacity_keys_for_node,
};
use crate::prelude::*;
use alloc::collections::{BTreeMap, BTreeSet};
use chelis_dag_core::dag::{DimExpr, DimInfo, NodeId, RiscOp};
use chelis_vocab::Repr;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageProvenance {
    RuntimeOwned,
    EntryBorrow,
    SharedView,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StorageId {
    pub key: u32,
}

impl StorageId {
    pub fn from_slot(slot: StorageSlotId) -> Result<Self, OwnershipError> {
        let key = u32::try_from(slot.index).map_err(|_| OwnershipError::LoweringInvariant {
            unit: "dag-storage".to_string(),
            detail: format!(
                "storage slot {} does not fit the opaque identity",
                slot.index
            ),
        })?;
        Ok(Self { key })
    }

    #[cfg(any(test, feature = "test-instrumentation"))]
    pub const fn from_index_for_test(key: u32) -> Self {
        Self { key }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StorageSlotId {
    pub index: usize,
}

impl StorageSlotId {
    const UNASSIGNED: Self = Self { index: usize::MAX };

    pub const fn index(self) -> usize {
        self.index
    }
}

#[derive(Clone, Debug)]
pub struct ExactStorageCapacity {
    key: CapacityKey,
    repr: Repr,
    allocation_bytes: Option<u64>,
}

impl ExactStorageCapacity {
    fn for_node(dag: DagAnalysisView<'_>, node: NodeId) -> Result<Self, OwnershipError> {
        let key = match capacity_key_for_node(dag, node).map_err(|error| {
            OwnershipError::DagStorageCapacity {
                node: node.0,
                detail: error.to_string(),
            }
        })? {
            CapacityKeyBuild::Exact(key) => key,
            CapacityKeyBuild::NotProven(reason) => {
                return Err(OwnershipError::DagStorageCapacity {
                    node: node.0,
                    detail: reason.to_string(),
                });
            }
        };
        let precision = dag
            .get(node)
            .expect("capacity node was checked above")
            .output_type
            .precision;
        let repr = precision
            .runtime_dtype()
            .map_err(|_| OwnershipError::DagStorageDType {
                node: node.0,
                dtype: precision.name(),
            })?
            .repr();
        let allocation_bytes = key.literal_allocation_bytes(repr).map_err(|error| {
            OwnershipError::DagStorageCapacity {
                node: node.0,
                detail: error.to_string(),
            }
        })?;
        Ok(Self {
            key,
            repr,
            allocation_bytes,
        })
    }

    fn proves_equal(&self, other: &Self) -> bool {
        self.repr == other.repr && self.key.prove_equal(&other.key).is_ok()
    }

    pub const fn repr(&self) -> Repr {
        self.repr
    }

    pub const fn allocation_bytes(&self) -> Option<u64> {
        self.allocation_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StoragePlacement {
    BorrowedEntry,
    InputMirror {
        slot: StorageSlotId,
        input_name: String,
    },
    OwnedSlot {
        slot: StorageSlotId,
    },
    SharedView {
        source: NodeId,
    },
    TerminalDrop {
        source: NodeId,
    },
    MaterializedStore {
        source: NodeId,
        slot: StorageSlotId,
    },
    StoreAlias {
        source: NodeId,
    },
    Skipped,
}

impl StoragePlacement {
    pub const fn slot(&self) -> Option<StorageSlotId> {
        match self {
            Self::InputMirror { slot, .. }
            | Self::OwnedSlot { slot }
            | Self::MaterializedStore { slot, .. } => Some(*slot),
            Self::BorrowedEntry
            | Self::SharedView { .. }
            | Self::TerminalDrop { .. }
            | Self::StoreAlias { .. }
            | Self::Skipped => None,
        }
    }
}

#[derive(Debug)]
pub struct StorageSlotPlan {
    id: StorageSlotId,
    capacity: ExactStorageCapacity,
    allocation_elements: DimExpr,
    first_owner: NodeId,
    last_use_index: usize,
}

impl StorageSlotPlan {
    pub const fn id(&self) -> StorageSlotId {
        self.id
    }

    pub const fn capacity(&self) -> &ExactStorageCapacity {
        &self.capacity
    }

    /// Renderable allocation extent paired with the exact capacity proof.
    pub const fn allocation_elements(&self) -> &DimExpr {
        &self.allocation_elements
    }

    pub const fn first_owner(&self) -> NodeId {
        self.first_owner
    }

    pub const fn last_use_index(&self) -> usize {
        self.last_use_index
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReuseProofCondition {
    OneLiveProgramOwner,
    UniqueDescriptorAndStorage,
    RuntimeOwnedAndWritable,
    NoLiveViewOrBorrow,
    ExactShapeDtypeCapacityAndOperation,
    TerminalUse,
}

impl ReuseProofCondition {
    #[cfg(any(test, feature = "test-instrumentation"))]
    pub const ALL: [Self; 6] = [
        Self::OneLiveProgramOwner,
        Self::UniqueDescriptorAndStorage,
        Self::RuntimeOwnedAndWritable,
        Self::NoLiveViewOrBorrow,
        Self::ExactShapeDtypeCapacityAndOperation,
        Self::TerminalUse,
    ];

    #[cfg(any(test, feature = "test-instrumentation"))]
    pub const fn name(self) -> &'static str {
        match self {
            Self::OneLiveProgramOwner => "one live program owner",
            Self::UniqueDescriptorAndStorage => "unique descriptor and storage",
            Self::RuntimeOwnedAndWritable => "runtime-owned and writable storage",
            Self::NoLiveViewOrBorrow => "no live view or borrow",
            Self::ExactShapeDtypeCapacityAndOperation => {
                "exact shape, dtype, capacity, and operation requirements"
            }
            Self::TerminalUse => "terminal source use",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactReuseProof(pub ());

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReuseCandidateFacts {
    pub provenance: StorageProvenance,
    pub one_live_program_owner: bool,
    pub unique_descriptor_and_storage: bool,
    pub writable: bool,
    pub no_live_view_or_borrow: bool,
    pub exact: Option<ExactReuseProof>,
    pub terminal_use: bool,
}

impl ReuseCandidateFacts {
    pub fn first_missing_condition(self) -> Option<ReuseProofCondition> {
        if !self.one_live_program_owner {
            return Some(ReuseProofCondition::OneLiveProgramOwner);
        }
        if !self.unique_descriptor_and_storage {
            return Some(ReuseProofCondition::UniqueDescriptorAndStorage);
        }
        if self.provenance != StorageProvenance::RuntimeOwned || !self.writable {
            return Some(ReuseProofCondition::RuntimeOwnedAndWritable);
        }
        if !self.no_live_view_or_borrow {
            return Some(ReuseProofCondition::NoLiveViewOrBorrow);
        }
        if self.exact.is_none() {
            return Some(ReuseProofCondition::ExactShapeDtypeCapacityAndOperation);
        }
        if !self.terminal_use {
            return Some(ReuseProofCondition::TerminalUse);
        }
        None
    }

    #[cfg(any(test, feature = "test-instrumentation"))]
    pub const fn complete_for_test(provenance: StorageProvenance) -> Self {
        Self {
            provenance,
            one_live_program_owner: true,
            unique_descriptor_and_storage: true,
            writable: true,
            no_live_view_or_borrow: true,
            exact: Some(ExactReuseProof(())),
            terminal_use: true,
        }
    }

    #[cfg(any(test, feature = "test-instrumentation"))]
    pub fn without_condition_for_test(mut self, condition: ReuseProofCondition) -> Self {
        match condition {
            ReuseProofCondition::OneLiveProgramOwner => self.one_live_program_owner = false,
            ReuseProofCondition::UniqueDescriptorAndStorage => {
                self.unique_descriptor_and_storage = false;
            }
            ReuseProofCondition::RuntimeOwnedAndWritable => self.writable = false,
            ReuseProofCondition::NoLiveViewOrBorrow => self.no_live_view_or_borrow = false,
            ReuseProofCondition::ExactShapeDtypeCapacityAndOperation => self.exact = None,
            ReuseProofCondition::TerminalUse => self.terminal_use = false,
        }
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReuseProofRejection {
    Missing(ReuseProofCondition),
}

#[derive(Debug, Eq, PartialEq)]
pub struct ReuseDecision {
    pub source: NodeId,
    pub storage: StorageId,
    pub consumer: NodeId,
}
impl ReuseDecision {
    pub fn calculate(
        source: NodeId,
        storage: StorageId,
        consumer: NodeId,
        facts: ReuseCandidateFacts,
    ) -> Result<Self, ReuseProofRejection> {
        if let Some(condition) = facts.first_missing_condition() {
            return Err(ReuseProofRejection::Missing(condition));
        }
        Ok(Self {
            source,
            storage,
            consumer,
        })
    }
}

#[derive(Clone, Copy)]
pub struct StoragePolicy {
    pub mirror_inputs: bool,
    pub shared_views: bool,
    pub skipped_keys: bool,
    pub alias_stores: bool,
    pub recurrence_tape: bool,
}
#[derive(Debug)]
struct OwnerRequirement {
    owner: NodeId,
    birth_index: usize,
    last_use_index: usize,
    capacity: ExactStorageCapacity,
    reusable_slot: bool,
    dedicated: bool,
}

struct AssignedSlots<R> {
    slots: Vec<StorageSlotPlan>,
    reusable: BTreeMap<NodeId, R>,
}

pub type BuiltStoragePlan<R> = (
    Box<[StoragePlacement]>,
    Box<[StorageSlotPlan]>,
    BTreeMap<NodeId, R>,
    LiveByteBound,
);

pub fn build_storage_plan<R>(
    dag: DagAnalysisView<'_>,
    policy: StoragePolicy,
    skipped: &chelis_unord::UnordSet<NodeId>,
    views: Option<&crate::contraction_views::ContractionViews>,
    mut mint: impl FnMut(ReuseDecision) -> R,
) -> Result<BuiltStoragePlan<R>, OwnershipError> {
    let mut placements = classify_nodes(dag, policy, skipped);
    if views.is_some() {
        check_inlined_product_storage(dag, &placements, views)?;
    }
    let owner_of = compute_owner_map(&placements);
    let destructively_dropped = destructively_dropped_owners(&placements, &owner_of);
    let mut requirements = owner_requirements(dag, &placements, &owner_of)?;
    extend_lifetimes(
        dag,
        policy,
        &placements,
        &owner_of,
        &mut requirements,
        views,
    );
    let AssignedSlots { slots, reusable } = assign_slots(
        dag,
        &mut placements,
        &owner_of,
        &requirements,
        &destructively_dropped,
        &mut mint,
    )?;
    let max_live_bytes = physical_live_byte_bound(&slots);
    Ok((
        placements.into_boxed_slice(),
        slots.into_boxed_slice(),
        reusable,
        max_live_bytes,
    ))
}
/// Every node an inlined product claims has the one storage decision its kernel needs.
/// The kernel's output owns a slot, each component it computes in place is skipped.
/// And each reshape it reads in place is a deferred view or owns a slot, never skipped as a component (chelis#3802).
fn check_inlined_product_storage(
    dag: DagAnalysisView<'_>,
    placements: &[StoragePlacement],
    views: Option<&crate::contraction_views::ContractionViews>,
) -> Result<(), OwnershipError> {
    let conflict = |node: NodeId, detail: &str| OwnershipError::DagStorageConflict {
        node: node.0,
        detail: detail.to_string(),
    };
    for product in dag.reduction_inlined_products() {
        let output = product.output();
        if placements[output.0].slot().is_none()
            || views.is_some_and(|views| views.is_deferred(output))
        {
            return Err(conflict(output, "an inlined product's output owns no slot"));
        }
        for node in product.skipped() {
            if !matches!(placements[node.0], StoragePlacement::Skipped) {
                return Err(conflict(node, "an inlined product's component is stored"));
            }
        }
        for reshape in product.view_reshapes.iter().flatten() {
            let deferred = views.is_some_and(|views| views.is_deferred(*reshape));
            if deferred != matches!(placements[reshape.0], StoragePlacement::Skipped) {
                return Err(conflict(
                    *reshape,
                    "an inlined product's view reshape is neither deferred nor stored",
                ));
            }
        }
    }
    Ok(())
}

fn classify_nodes(
    dag: DagAnalysisView<'_>,
    policy: StoragePolicy,
    skipped: &chelis_unord::UnordSet<NodeId>,
) -> Vec<StoragePlacement> {
    let mut placements = Vec::with_capacity(dag.len());
    let mut first_load_by_name = BTreeMap::<String, NodeId>::new();
    for node in dag.nodes() {
        let placement = if skipped.contains(&node.id) {
            StoragePlacement::Skipped
        } else {
            match &node.op {
                RiscOp::Load { name } => {
                    if !policy.mirror_inputs {
                        StoragePlacement::BorrowedEntry
                    } else {
                        if let Some(source) = first_load_by_name.get(name.as_str()) {
                            StoragePlacement::SharedView { source: *source }
                        } else {
                            first_load_by_name.insert(name.as_str().to_string(), node.id);
                            StoragePlacement::InputMirror {
                                slot: StorageSlotId::UNASSIGNED,
                                input_name: name.as_str().to_string(),
                            }
                        }
                    }
                }
                // A HIP input can have arbitrary checked strides.
                // Reshape must materialize logical order, so its bytes and lifetime belong to the shared plan just as they do on the C lane.
                RiscOp::Reshape { .. } => StoragePlacement::OwnedSlot {
                    slot: StorageSlotId::UNASSIGNED,
                },
                RiscOp::Permute { .. } | RiscOp::Expand { .. } | RiscOp::Stride { .. } => {
                    if policy.shared_views {
                        StoragePlacement::SharedView {
                            source: node.inputs[0],
                        }
                    } else {
                        StoragePlacement::OwnedSlot {
                            slot: StorageSlotId::UNASSIGNED,
                        }
                    }
                }
                // A derived or joined key is an ordinary key tensor on the C lane.
                // The HIP lane computes rank-0 derivations while it emits, and refuses the rest, so they take no device storage.
                RiscOp::KeyFromSeed
                | RiscOp::Split { .. }
                | RiscOp::FoldIn
                | RiscOp::SplitN { .. }
                | RiscOp::KeySelect => {
                    if policy.skipped_keys {
                        StoragePlacement::Skipped
                    } else {
                        StoragePlacement::OwnedSlot {
                            slot: StorageSlotId::UNASSIGNED,
                        }
                    }
                }
                RiscOp::Drop => StoragePlacement::TerminalDrop {
                    source: node.inputs[0],
                },
                RiscOp::Store { .. } => {
                    if policy.alias_stores {
                        StoragePlacement::StoreAlias {
                            source: node.inputs[0],
                        }
                    } else {
                        StoragePlacement::MaterializedStore {
                            source: node.inputs[0],
                            slot: StorageSlotId::UNASSIGNED,
                        }
                    }
                }
                RiscOp::ListMapCapture { .. }
                | RiscOp::OrderedAdjointSum { .. }
                | RiscOp::Iota
                | RiscOp::Const { .. }
                | RiscOp::ConstTensor { .. }
                | RiscOp::Shape { .. }
                | RiscOp::ExtentWitness { .. }
                | RiscOp::CheckedExtent { .. }
                | RiscOp::CheckedUnitAxis { .. }
                | RiscOp::Add
                | RiscOp::Sub
                | RiscOp::Mul
                | RiscOp::Div
                | RiscOp::Pow
                | RiscOp::FloorDiv
                | RiscOp::TruncDiv
                | RiscOp::Mod
                | RiscOp::Bitwise(_)
                | RiscOp::FloatClassify(_)
                | RiscOp::Compare(_)
                | RiscOp::Logical(_)
                | RiscOp::Where
                | RiscOp::GuardedFail { .. }
                | RiscOp::MaxElem
                | RiscOp::MinElem
                | RiscOp::ExtremaAdjoint { .. }
                | RiscOp::Relu
                | RiscOp::Softmax { .. }
                | RiscOp::ReluAdjoint
                | RiscOp::Neg
                | RiscOp::Recip
                | RiscOp::Exp
                | RiscOp::Log
                | RiscOp::Sin
                | RiscOp::Sqrt
                | RiscOp::Cos
                | RiscOp::Tan
                | RiscOp::Atan
                | RiscOp::Tanh
                | RiscOp::Erf
                | RiscOp::Erfc
                | RiscOp::Abs
                | RiscOp::Floor
                | RiscOp::Ceil
                | RiscOp::Round
                | RiscOp::UniformLike
                | RiscOp::Dropout
                | RiscOp::DropoutReplay
                | RiscOp::UniformBoundAdjoint { .. }
                | RiscOp::Copy
                | RiscOp::Cumsum { .. }
                | RiscOp::Sum { .. }
                | RiscOp::Count { .. }
                | RiscOp::MaxReduce { .. }
                | RiscOp::MinReduce { .. }
                | RiscOp::ProdReduce { .. }
                | RiscOp::ReduceWindow { .. }
                | RiscOp::ReduceWindowGrad { .. }
                | RiscOp::Argmax { .. }
                | RiscOp::Argmin { .. }
                | RiscOp::OneHot { .. }
                | RiscOp::Realize
                | RiscOp::Cast { .. }
                | RiscOp::NamedCast { .. }
                | RiscOp::FusedElem { .. }
                | RiscOp::Pad { .. }
                | RiscOp::Shrink { .. }
                | RiscOp::BlasMatmul { .. }
                | RiscOp::Gather { .. }
                | RiscOp::ScatterAdd { .. }
                | RiscOp::Scatter { .. }
                | RiscOp::ScatterElements { .. }
                | RiscOp::RuntimeRecurrence(_)
                | RiscOp::RuntimeRecurrenceAdjoint { .. } => StoragePlacement::OwnedSlot {
                    slot: StorageSlotId::UNASSIGNED,
                },
            }
        };
        placements.push(placement);
    }
    placements
}

fn compute_owner_map(placements: &[StoragePlacement]) -> Vec<Option<NodeId>> {
    let mut owners = vec![None; placements.len()];
    for (index, placement) in placements.iter().enumerate() {
        let node = NodeId(index);
        owners[index] = match placement {
            StoragePlacement::InputMirror { .. }
            | StoragePlacement::OwnedSlot { .. }
            | StoragePlacement::MaterializedStore { .. } => Some(node),
            StoragePlacement::SharedView { source } | StoragePlacement::StoreAlias { source } => {
                owners[source.0]
            }
            StoragePlacement::BorrowedEntry
            | StoragePlacement::TerminalDrop { .. }
            | StoragePlacement::Skipped => None,
        };
    }
    owners
}

fn destructively_dropped_owners(
    placements: &[StoragePlacement],
    owner_of: &[Option<NodeId>],
) -> BTreeSet<NodeId> {
    placements
        .iter()
        .filter_map(|placement| match placement {
            StoragePlacement::TerminalDrop { source } => owner_of[source.0],
            _ => None,
        })
        .collect()
}

fn owner_requirements(
    dag: DagAnalysisView<'_>,
    placements: &[StoragePlacement],
    owner_of: &[Option<NodeId>],
) -> Result<BTreeMap<NodeId, OwnerRequirement>, OwnershipError> {
    let mut result = BTreeMap::new();
    for node in dag.nodes() {
        let (reusable_slot, dedicated) = match placements[node.id.0] {
            StoragePlacement::OwnedSlot { .. } => (true, false),
            StoragePlacement::InputMirror { .. } => (false, false),
            StoragePlacement::MaterializedStore { .. } => (false, true),
            StoragePlacement::BorrowedEntry
            | StoragePlacement::SharedView { .. }
            | StoragePlacement::TerminalDrop { .. }
            | StoragePlacement::StoreAlias { .. }
            | StoragePlacement::Skipped => continue,
        };
        debug_assert_eq!(owner_of[node.id.0], Some(node.id));
        result.insert(
            node.id,
            OwnerRequirement {
                owner: node.id,
                birth_index: node.id.0,
                last_use_index: node.id.0,
                capacity: ExactStorageCapacity::for_node(dag, node.id)?,
                reusable_slot,
                dedicated,
            },
        );
    }
    Ok(result)
}

fn extend_lifetimes(
    dag: DagAnalysisView<'_>,
    policy: StoragePolicy,
    placements: &[StoragePlacement],
    owner_of: &[Option<NodeId>],
    requirements: &mut BTreeMap<NodeId, OwnerRequirement>,
    views: Option<&crate::contraction_views::ContractionViews>,
) {
    let epilogue = dag.len();
    for node in dag.nodes() {
        if matches!(placements[node.id.0], StoragePlacement::Skipped)
            && !views.is_some_and(|views| views.is_deferred(node.id))
        {
            continue;
        }
        // Deferred storage omits only the payload.
        // Its original operation still reads runtime bounds, shape/claim witnesses and activation at this position, before later consumers read the recipe's leaves.
        let mut effective_inputs = match &node.op {
            RiscOp::Drop => match placements[node.id.0] {
                StoragePlacement::TerminalDrop { source } => vec![source],
                _ => node.inputs.clone(),
            },
            // A node computing a skipped input inline reads, at its own position, everything that input would read.
            _ => read_through_skipped(dag, placements, &node.inputs, views),
        };
        // Ownership borrows shape dependencies as well as value inputs.
        // A result-claim witness stores the canonical scalar the producer reads.
        // reusing its allocation earlier changes the obligation itself.
        effective_inputs.extend(node.shape_deps.iter().copied());
        effective_inputs.extend(node.result_claim_deps.iter().copied());
        // A node reads its activation to decide whether it checks.
        effective_inputs.extend(node.owner.activation);
        for input in effective_inputs {
            if let Some(owner) = owner_of[input.0]
                && let Some(requirement) = requirements.get_mut(&owner)
            {
                requirement.last_use_index = requirement.last_use_index.max(node.id.0);
            }
        }
        if policy.recurrence_tape
            && let RiscOp::RuntimeRecurrence(recurrence) = &node.op
        {
            // The C callback returns every node into its tape.
            // A Load of the seed or a lexical capture can retain the parent tensor through tape cleanup.
            // Even after the recurrence node's logical use.
            // The source and columns are gathered into fresh per-row values.
            let tape_aliases = core::iter::once(node.inputs[1]).chain(
                node.inputs[recurrence.capture_input_offset()..]
                    .iter()
                    .copied(),
            );
            for input in tape_aliases {
                if let Some(owner) = owner_of[input.0]
                    && let Some(requirement) = requirements.get_mut(&owner)
                {
                    requirement.last_use_index = epilogue;
                }
            }
            // A fold returns the final taped state, or the seed at zero steps.
            // A scan materializes an independent stacked result.
            if !recurrence.scan
                && let Some(owner) = owner_of[node.id.0]
                && let Some(requirement) = requirements.get_mut(&owner)
            {
                requirement.last_use_index = epilogue;
            }
        }
    }
    for root in dag.roots() {
        if let Some(owner) = owner_of[root.0]
            && let Some(requirement) = requirements.get_mut(&owner)
        {
            requirement.last_use_index = epilogue;
        }
    }
    // The recipe planner propagates deferred reads in one reverse pass.
    // These are payload borrows even when the original node only carried metadata at the later consumer.
    // a source slot cannot be repurposed.
    if let Some(views) = views {
        for node in dag.nodes() {
            if let Some(last) = views.base_last_read(node.id)
                && let Some(owner) = owner_of[node.id.0]
                && let Some(requirement) = requirements.get_mut(&owner)
            {
                requirement.last_use_index = requirement.last_use_index.max(last);
            }
        }
    }
}

/// `inputs`, with each skipped node replaced by its own reads, transitively.
fn read_through_skipped(
    dag: DagAnalysisView<'_>,
    placements: &[StoragePlacement],
    inputs: &[NodeId],
    views: Option<&crate::contraction_views::ContractionViews>,
) -> Vec<NodeId> {
    let mut reads = Vec::new();
    let mut pending = inputs.iter().rev().copied().collect::<Vec<_>>();
    while let Some(input) = pending.pop() {
        if views.is_some_and(|views| views.is_deferred(input)) {
            // Accounted for by the reverse-topological base-retirement pass.
            continue;
        }
        if !matches!(placements[input.0], StoragePlacement::Skipped) {
            reads.push(input);
            continue;
        }
        let node = dag.get(input).expect("verified skipped input names a node");
        pending.extend(
            node.inputs
                .iter()
                .chain(&node.shape_deps)
                .chain(&node.result_claim_deps)
                .chain(&node.owner.activation)
                .rev()
                .copied(),
        );
    }
    reads
}

fn assign_slots<R>(
    dag: DagAnalysisView<'_>,
    placements: &mut [StoragePlacement],
    owner_of: &[Option<NodeId>],
    requirements: &BTreeMap<NodeId, OwnerRequirement>,
    destructively_dropped: &BTreeSet<NodeId>,
    mint: &mut impl FnMut(ReuseDecision) -> R,
) -> Result<AssignedSlots<R>, OwnershipError> {
    let mut slots = Vec::<StorageSlotPlan>::new();
    let mut availability = Vec::<usize>::new();
    let mut future_reuse_allowed = Vec::<bool>::new();
    let mut owner_to_slot = BTreeMap::<NodeId, StorageSlotId>::new();
    let mut reusable = BTreeMap::new();

    for requirement in requirements.values() {
        let exact_in_place =
            exact_in_place_candidate(dag, placements, owner_of, requirements, requirement)?;
        let mut selected = None;
        let mut token_facts = None;
        if let Some((source, facts)) = exact_in_place
            && let Some(slot) = owner_to_slot.get(&source).copied()
        {
            selected = Some(slot);
            token_facts = Some((source, facts));
        }
        if selected.is_none() && requirement.reusable_slot && !requirement.dedicated {
            selected = slots.iter().find_map(|slot| {
                let index = slot.id.index;
                (future_reuse_allowed[index]
                    && availability[index] < requirement.birth_index
                    && slot.capacity.proves_equal(&requirement.capacity))
                .then_some(slot.id)
            });
        }
        let slot = match selected {
            Some(slot) => slot,
            None => {
                let slot = StorageSlotId { index: slots.len() };
                slots.push(StorageSlotPlan {
                    id: slot,
                    capacity: requirement.capacity.clone(),
                    allocation_elements: allocation_elements_for_node(dag, requirement.owner),
                    first_owner: requirement.owner,
                    last_use_index: requirement.last_use_index,
                });
                availability.push(usize::MIN);
                future_reuse_allowed.push(
                    requirement.reusable_slot
                        && !requirement.dedicated
                        && !destructively_dropped.contains(&requirement.owner),
                );
                slot
            }
        };
        let index = slot.index;
        availability[index] = requirement.last_use_index;
        slots[index].last_use_index = slots[index].last_use_index.max(requirement.last_use_index);
        future_reuse_allowed[index] = requirement.reusable_slot
            && !requirement.dedicated
            && !destructively_dropped.contains(&requirement.owner);
        owner_to_slot.insert(requirement.owner, slot);
        assign_placement_slot(&mut placements[requirement.owner.0], slot);

        if let Some((source, facts)) = token_facts {
            let decision = ReuseDecision::calculate(
                source,
                StorageId::from_slot(slot)?,
                requirement.owner,
                facts,
            )
            .map_err(|rejection| OwnershipError::LoweringInvariant {
                unit: "dag-storage".to_string(),
                detail: format!(
                    "selected reuse n{} -> n{} without {:?}",
                    source.0, requirement.owner.0, rejection
                ),
            })?;
            if reusable.insert(requirement.owner, mint(decision)).is_some() {
                return Err(OwnershipError::LoweringInvariant {
                    unit: "dag-storage".to_string(),
                    detail: format!(
                        "consumer n{} received two reuse tokens",
                        requirement.owner.0
                    ),
                });
            }
        }
    }
    Ok(AssignedSlots { slots, reusable })
}

fn allocation_elements_for_node(dag: DagAnalysisView<'_>, node: NodeId) -> DimExpr {
    let tensor = &dag
        .get(node)
        .expect("storage requirement names a verified node")
        .output_type;
    if tensor.dims.is_empty() {
        return DimExpr::Concrete(1);
    }
    tensor
        .dims
        .iter()
        .map(|dim| match dim {
            DimInfo::Lit(value) | DimInfo::Named(_, Some(value)) => DimExpr::Concrete(*value),
            DimInfo::Named(name, None) => DimExpr::Sym(name.clone()),
        })
        .reduce(|left, right| DimExpr::Mul(Box::new(left), Box::new(right)))
        .unwrap_or(DimExpr::Concrete(1))
}

fn assign_placement_slot(placement: &mut StoragePlacement, slot: StorageSlotId) {
    match placement {
        StoragePlacement::InputMirror { slot: assigned, .. }
        | StoragePlacement::OwnedSlot { slot: assigned }
        | StoragePlacement::MaterializedStore { slot: assigned, .. } => *assigned = slot,
        StoragePlacement::BorrowedEntry
        | StoragePlacement::SharedView { .. }
        | StoragePlacement::TerminalDrop { .. }
        | StoragePlacement::StoreAlias { .. }
        | StoragePlacement::Skipped => unreachable!("only allocating placements receive slots"),
    }
}

fn exact_in_place_candidate(
    dag: DagAnalysisView<'_>,
    placements: &[StoragePlacement],
    owner_of: &[Option<NodeId>],
    requirements: &BTreeMap<NodeId, OwnerRequirement>,
    consumer_requirement: &OwnerRequirement,
) -> Result<Option<(NodeId, ReuseCandidateFacts)>, OwnershipError> {
    let consumer = dag
        .get(consumer_requirement.owner)
        .expect("storage requirement names a verified node");
    let Some(source) = consumer.reusable_input else {
        return Ok(None);
    };
    let source_requirement = requirements.get(&source);
    let aliases_storage = owner_of
        .iter()
        .enumerate()
        .any(|(node, owner)| node != source.0 && *owner == Some(source));
    let source_occurrences = consumer
        .inputs
        .iter()
        .filter(|input| **input == source)
        .count();
    let total_consumers = dag
        .nodes()
        .iter()
        .flat_map(|node| node.inputs.iter())
        .filter(|input| **input == source)
        .count()
        + dag.roots().iter().filter(|root| **root == source).count();
    let exact = match source_requirement {
        Some(source_requirement)
            if matches!(consumer.op, RiscOp::FusedElem { .. })
                && source_requirement
                    .capacity
                    .proves_equal(&consumer_requirement.capacity)
                && exact_shape_equal(dag, source, consumer.id)? =>
        {
            Some(ExactReuseProof(()))
        }
        _ => None,
    };
    let facts = ReuseCandidateFacts {
        provenance: match placements.get(source.0) {
            Some(StoragePlacement::OwnedSlot { .. }) => StorageProvenance::RuntimeOwned,
            Some(StoragePlacement::BorrowedEntry | StoragePlacement::InputMirror { .. }) => {
                StorageProvenance::EntryBorrow
            }
            Some(StoragePlacement::SharedView { .. } | StoragePlacement::StoreAlias { .. }) => {
                StorageProvenance::SharedView
            }
            Some(
                StoragePlacement::TerminalDrop { .. }
                | StoragePlacement::MaterializedStore { .. }
                | StoragePlacement::Skipped,
            )
            | None => StorageProvenance::SharedView,
        },
        one_live_program_owner: source_requirement.is_some() && owner_of[source.0] == Some(source),
        unique_descriptor_and_storage: !aliases_storage,
        writable: matches!(
            placements.get(source.0),
            Some(StoragePlacement::OwnedSlot { .. })
        ),
        no_live_view_or_borrow: source_occurrences == 1 && !aliases_storage,
        exact,
        terminal_use: source_requirement
            .is_some_and(|requirement| requirement.last_use_index == consumer.id.0)
            && total_consumers == 1,
    };
    if facts.first_missing_condition().is_some() {
        Ok(None)
    } else {
        Ok(Some((source, facts)))
    }
}

fn exact_shape_equal(
    dag: DagAnalysisView<'_>,
    source: NodeId,
    consumer: NodeId,
) -> Result<bool, OwnershipError> {
    let source_keys = shape_capacity_keys_for_node(dag, source).map_err(|error| {
        OwnershipError::DagStorageCapacity {
            node: source.0,
            detail: error.to_string(),
        }
    })?;
    let consumer_keys = shape_capacity_keys_for_node(dag, consumer).map_err(|error| {
        OwnershipError::DagStorageCapacity {
            node: consumer.0,
            detail: error.to_string(),
        }
    })?;
    if source_keys.len() != consumer_keys.len() {
        return Ok(false);
    }
    Ok(source_keys
        .iter()
        .zip(&consumer_keys)
        .all(|(source, consumer)| match (source, consumer) {
            (CapacityKeyBuild::Exact(source), CapacityKeyBuild::Exact(consumer)) => {
                source.prove_equal(consumer).is_ok()
            }
            _ => false,
        }))
}

fn physical_live_byte_bound(slots: &[StorageSlotPlan]) -> LiveByteBound {
    // Logical death lets reuse.
    // it does not release physical storage.
    // HIP retains its entire slot pool through cleanup, and C retains descriptors unless explicitly dropped.
    // Count every distinct slot once, including dead gaps.
    // This is a concrete upper bound, conservative for early C Drops, not a claim that every lane necessarily observes the same exact peak.
    //
    exact_physical_live_bytes(slots).unwrap_or_else(|bound| bound)
}

/// The exact physical total, or the non-exact bound that replaces it.
/// A total beyond u64 has no finite representable bound and is `Unbounded`, never an error.
/// Each slot's allocation keeps its own checked admission (chelis#3418).
/// `Unbounded` dominates `Unknown`, as in the verifier's composition, so the answer does not depend on slot order.
fn exact_physical_live_bytes(slots: &[StorageSlotPlan]) -> Result<LiveByteBound, LiveByteBound> {
    let mut total = 0u64;
    let mut unknown = false;
    for slot in slots {
        let Some(bytes) = slot.capacity.allocation_bytes else {
            unknown = true;
            continue;
        };
        total = total
            .checked_add(bytes)
            // Past u64 no finite bound is representable.
            .ok_or(LiveByteBound::Unbounded)?;
    }
    if unknown {
        return Err(LiveByteBound::Unknown);
    }
    Ok(LiveByteBound::Exact(total))
}

#[cfg(test)]
mod tests;
