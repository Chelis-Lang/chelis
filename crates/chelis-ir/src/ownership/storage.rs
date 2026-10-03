//! Shared exact-capacity storage planning for the C and HIP lanes.

use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;

use chelis_vocab::Repr;

use crate::capacity_key::{
    CapacityKey, CapacityKeyBuild, capacity_key_for_node, shape_capacity_keys_for_node,
};
use crate::dag::{DimExpr, DimInfo, NodeId, RiscOp};

use super::{LiveByteBound, OwnershipError, VerifiedDagProgram, VerifiedDagView};

#[derive(Debug)]
pub enum CStorageLane {}

#[derive(Debug)]
pub enum HipStorageLane {}

mod sealed {
    pub trait StorageLane {}
}

pub trait StorageLane: sealed::StorageLane {}

impl sealed::StorageLane for CStorageLane {}
impl StorageLane for CStorageLane {}
impl sealed::StorageLane for HipStorageLane {}
impl StorageLane for HipStorageLane {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StorageLaneKind {
    C,
    Hip,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StorageProvenance {
    RuntimeOwned,
    EntryBorrow,
    SharedView,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct StorageId {
    key: u32,
}

impl StorageId {
    fn from_slot(slot: StorageSlotId) -> Result<Self, OwnershipError> {
        let key = u32::try_from(slot.index).map_err(|_| OwnershipError::LoweringInvariant {
            unit: "dag-storage".to_string(),
            detail: format!(
                "storage slot {} does not fit the opaque identity",
                slot.index
            ),
        })?;
        Ok(Self { key })
    }

    #[cfg(test)]
    const fn from_index_for_test(key: u32) -> Self {
        Self { key }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StorageSlotId {
    index: usize,
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
    fn for_node(dag: VerifiedDagView<'_>, node: NodeId) -> Result<Self, OwnershipError> {
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
pub(crate) enum ReuseProofCondition {
    OneLiveProgramOwner,
    UniqueDescriptorAndStorage,
    RuntimeOwnedAndWritable,
    NoLiveViewOrBorrow,
    ExactShapeDtypeCapacityAndOperation,
    TerminalUse,
}

impl ReuseProofCondition {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 6] = [
        Self::OneLiveProgramOwner,
        Self::UniqueDescriptorAndStorage,
        Self::RuntimeOwnedAndWritable,
        Self::NoLiveViewOrBorrow,
        Self::ExactShapeDtypeCapacityAndOperation,
        Self::TerminalUse,
    ];

    #[cfg(test)]
    pub(crate) const fn name(self) -> &'static str {
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
struct ExactReuseProof(());

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReuseCandidateFacts {
    provenance: StorageProvenance,
    one_live_program_owner: bool,
    unique_descriptor_and_storage: bool,
    writable: bool,
    no_live_view_or_borrow: bool,
    exact: Option<ExactReuseProof>,
    terminal_use: bool,
}

impl ReuseCandidateFacts {
    fn first_missing_condition(self) -> Option<ReuseProofCondition> {
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

    #[cfg(test)]
    const fn complete_for_test(provenance: StorageProvenance) -> Self {
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

    #[cfg(test)]
    fn without_condition_for_test(mut self, condition: ReuseProofCondition) -> Self {
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
enum ReuseProofRejection {
    Missing(ReuseProofCondition),
}

/// Linear authority for one exact program-owned in-place reuse.
///
/// ```compile_fail
/// use chelis_ir::ownership::ReusableOwnedStorage;
/// fn consume(_: ReusableOwnedStorage) {}
/// fn duplicate(token: ReusableOwnedStorage) { consume(token); consume(token); }
/// ```
///
/// ```compile_fail
/// use chelis_ir::ownership::ReusableOwnedStorage;
/// fn inspect(token: &ReusableOwnedStorage) { let _ = token.storage; }
/// ```
///
/// ```compile_fail
/// use chelis_ir::ownership::ReusableOwnedStorage;
/// fn require_clone<T: Clone>() {}
/// fn duplicate() { require_clone::<ReusableOwnedStorage>(); }
/// ```
///
/// ```compile_fail
/// use chelis_ir::ownership::ReusableOwnedStorage;
/// fn require_copy<T: Copy>() {}
/// fn duplicate() { require_copy::<ReusableOwnedStorage>(); }
/// ```
///
/// ```compile_fail
/// # use chelis_ir::dag::NodeId;
/// use chelis_ir::ownership::ReusableOwnedStorage;
/// fn forge(source: NodeId, consumer: NodeId) -> ReusableOwnedStorage {
///     ReusableOwnedStorage { source, storage: todo!(), consumer }
/// }
/// ```
#[derive(Debug, Eq, PartialEq)]
pub struct ReusableOwnedStorage {
    source: NodeId,
    storage: StorageId,
    consumer: NodeId,
}

impl ReusableOwnedStorage {
    pub const fn source(&self) -> NodeId {
        self.source
    }

    pub const fn consumer(&self) -> NodeId {
        self.consumer
    }

    fn mint(
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

    #[cfg(test)]
    fn mint_for_test(
        source: NodeId,
        storage: StorageId,
        consumer: NodeId,
        facts: ReuseCandidateFacts,
    ) -> Result<Self, ReuseProofRejection> {
        Self::mint(source, storage, consumer, facts)
    }

    #[cfg(test)]
    const fn storage_for_test(&self) -> StorageId {
        self.storage
    }
}

#[derive(Debug)]
pub struct VerifiedStoragePlan<L: StorageLane> {
    program: VerifiedDagProgram,
    placements: Box<[StoragePlacement]>,
    slots: Box<[StorageSlotPlan]>,
    reusable: BTreeMap<NodeId, ReusableOwnedStorage>,
    consumed_reuse: BTreeSet<NodeId>,
    max_live_bytes: LiveByteBound,
    lane: PhantomData<L>,
}

/// Borrowing form used when a verified DAG is owned by a verified host
/// program. It carries the exact parent-owned DAG view and the same linear
/// token set; standalone backend entry points use [`VerifiedStoragePlan`].
pub struct VerifiedStorageLayout<'a, L: StorageLane> {
    dag: VerifiedDagView<'a>,
    placements: Box<[StoragePlacement]>,
    slots: Box<[StorageSlotPlan]>,
    reusable: BTreeMap<NodeId, ReusableOwnedStorage>,
    consumed_reuse: BTreeSet<NodeId>,
    max_live_bytes: LiveByteBound,
    lane: PhantomData<L>,
}

impl<L: StorageLane> VerifiedStoragePlan<L> {
    pub fn emission(&self) -> VerifiedDagView<'_> {
        self.program.emission()
    }

    pub fn placement(&self, node: NodeId) -> Option<&StoragePlacement> {
        self.placements.get(node.0)
    }

    pub fn placements(&self) -> &[StoragePlacement] {
        &self.placements
    }

    pub fn slots(&self) -> &[StorageSlotPlan] {
        &self.slots
    }

    pub fn slot(&self, id: StorageSlotId) -> Option<&StorageSlotPlan> {
        self.slots.get(id.index)
    }

    pub fn slot_for_node(&self, node: NodeId) -> Option<StorageSlotId> {
        self.placement(node).and_then(StoragePlacement::slot)
    }

    pub const fn max_live_bytes(&self) -> LiveByteBound {
        self.max_live_bytes
    }

    pub fn take_reuse_for(
        &mut self,
        consumer: NodeId,
    ) -> Result<Option<ReusableOwnedStorage>, OwnershipError> {
        if self.consumed_reuse.contains(&consumer) {
            return Err(OwnershipError::DagStorageReuseTaken {
                consumer: consumer.0,
            });
        }
        let token = self.reusable.remove(&consumer);
        if token.is_some() {
            self.consumed_reuse.insert(consumer);
        }
        Ok(token)
    }
}

impl<'a, L: StorageLane> VerifiedStorageLayout<'a, L> {
    pub const fn emission(&self) -> VerifiedDagView<'a> {
        self.dag
    }

    pub fn placements(&self) -> &[StoragePlacement] {
        &self.placements
    }

    pub fn slots(&self) -> &[StorageSlotPlan] {
        &self.slots
    }

    pub const fn max_live_bytes(&self) -> LiveByteBound {
        self.max_live_bytes
    }

    pub fn take_reuse_for(
        &mut self,
        consumer: NodeId,
    ) -> Result<Option<ReusableOwnedStorage>, OwnershipError> {
        if self.consumed_reuse.contains(&consumer) {
            return Err(OwnershipError::DagStorageReuseTaken {
                consumer: consumer.0,
            });
        }
        let token = self.reusable.remove(&consumer);
        if token.is_some() {
            self.consumed_reuse.insert(consumer);
        }
        Ok(token)
    }
}

pub fn plan_c_storage(
    program: VerifiedDagProgram,
) -> Result<VerifiedStoragePlan<CStorageLane>, OwnershipError> {
    plan_storage(program, StorageLaneKind::C)
}

pub fn plan_hip_storage(
    program: VerifiedDagProgram,
) -> Result<VerifiedStoragePlan<HipStorageLane>, OwnershipError> {
    plan_storage(program, StorageLaneKind::Hip)
}

pub fn plan_c_storage_layout(
    dag: VerifiedDagView<'_>,
) -> Result<VerifiedStorageLayout<'_, CStorageLane>, OwnershipError> {
    let (placements, slots, reusable, max_live_bytes) =
        build_storage_plan(dag, StorageLaneKind::C)?;
    Ok(VerifiedStorageLayout {
        dag,
        placements,
        slots,
        reusable,
        consumed_reuse: BTreeSet::new(),
        max_live_bytes,
        lane: PhantomData,
    })
}

fn plan_storage<L: StorageLane>(
    program: VerifiedDagProgram,
    lane: StorageLaneKind,
) -> Result<VerifiedStoragePlan<L>, OwnershipError> {
    let (placements, slots, reusable, max_live_bytes) = {
        let dag = program.emission();
        build_storage_plan(dag, lane)?
    };
    Ok(VerifiedStoragePlan {
        program,
        placements,
        slots,
        reusable,
        consumed_reuse: BTreeSet::new(),
        max_live_bytes,
        lane: PhantomData,
    })
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

struct AssignedSlots {
    slots: Vec<StorageSlotPlan>,
    reusable: BTreeMap<NodeId, ReusableOwnedStorage>,
}

type BuiltStoragePlan = (
    Box<[StoragePlacement]>,
    Box<[StorageSlotPlan]>,
    BTreeMap<NodeId, ReusableOwnedStorage>,
    LiveByteBound,
);

fn build_storage_plan(
    dag: VerifiedDagView<'_>,
    lane: StorageLaneKind,
) -> Result<BuiltStoragePlan, OwnershipError> {
    let mut skipped = dag.reduction_inlined_fused_elems();
    if lane == StorageLaneKind::Hip {
        skipped.extend(hip_emission_literals(dag));
    }
    let mut placements = classify_nodes(dag, lane, &skipped);
    let owner_of = compute_owner_map(&placements);
    let destructively_dropped = destructively_dropped_owners(&placements, &owner_of);
    let mut requirements = owner_requirements(dag, &placements, &owner_of)?;
    extend_lifetimes(dag, &placements, &owner_of, &mut requirements);
    let AssignedSlots { slots, reusable } = assign_slots(
        dag,
        &mut placements,
        &owner_of,
        &requirements,
        &destructively_dropped,
    )?;
    let max_live_bytes = physical_live_byte_bound(&slots)?;
    Ok((
        placements.into_boxed_slice(),
        slots.into_boxed_slice(),
        reusable,
        max_live_bytes,
    ))
}

/// The constants the HIP emitter reads only as literals while it emits a draw:
/// a draw key's seed and controls, a key derivation's seed or index, and a
/// `uniform_like`'s bounds, which the emitter folds into the key it computes
/// and the kernel's arguments. Such a
/// constant has no device value, so it takes no slot and no lifetime. A
/// constant any other operation, dependency or root reads keeps its storage.
fn hip_emission_literals(dag: VerifiedDagView<'_>) -> Vec<NodeId> {
    // `None` until a read is seen; then whether every read is a literal one.
    let mut literal_only = vec![None::<bool>; dag.len()];
    let mut read = |input: NodeId, literal: bool| {
        if let Some(entry) = literal_only.get_mut(input.0) {
            *entry = Some(entry.unwrap_or(true) && literal);
        }
    };
    for node in dag.nodes() {
        for (slot, input) in node.inputs.iter().enumerate() {
            let literal = match node.op {
                RiscOp::KeyFromSeed => true,
                RiscOp::FoldIn => slot == 1,
                RiscOp::UniformLike => matches!(slot, 1 | 2),
                _ => false,
            };
            read(*input, literal);
        }
        for dependency in node
            .shape_deps
            .iter()
            .chain(&node.result_claim_deps)
            .chain(&node.owner.activation)
        {
            read(*dependency, false);
        }
    }
    for root in dag.roots() {
        read(*root, false);
    }
    dag.nodes()
        .iter()
        .filter(|node| {
            matches!(node.op, RiscOp::Const { .. }) && literal_only[node.id.0] == Some(true)
        })
        .map(|node| node.id)
        .collect()
}

fn classify_nodes(
    dag: VerifiedDagView<'_>,
    lane: StorageLaneKind,
    skipped: &chelis_unord::UnordSet<NodeId>,
) -> Vec<StoragePlacement> {
    let mut placements = Vec::with_capacity(dag.len());
    let mut first_load_by_name = BTreeMap::<String, NodeId>::new();
    for node in dag.nodes() {
        let placement = if skipped.contains(&node.id) {
            StoragePlacement::Skipped
        } else {
            match &node.op {
                RiscOp::Load { name } => match lane {
                    StorageLaneKind::C => StoragePlacement::BorrowedEntry,
                    StorageLaneKind::Hip => {
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
                },
                // A HIP input can have arbitrary checked strides. Reshape must
                // materialize logical order, so its bytes and lifetime belong
                // to the shared plan just as they do on the C lane.
                RiscOp::Reshape { .. } => StoragePlacement::OwnedSlot {
                    slot: StorageSlotId::UNASSIGNED,
                },
                RiscOp::Permute { .. } | RiscOp::Expand { .. } | RiscOp::Stride { .. } => {
                    match lane {
                        StorageLaneKind::C => StoragePlacement::OwnedSlot {
                            slot: StorageSlotId::UNASSIGNED,
                        },
                        StorageLaneKind::Hip => StoragePlacement::SharedView {
                            source: node.inputs[0],
                        },
                    }
                }
                // A derived or joined key is an ordinary key tensor on the C
                // lane. The HIP lane computes rank-0 derivations while it
                // emits, and refuses the rest, so they take no device
                // storage.
                RiscOp::KeyFromSeed
                | RiscOp::Split { .. }
                | RiscOp::FoldIn
                | RiscOp::SplitN { .. }
                | RiscOp::KeySelect => match lane {
                    StorageLaneKind::C => StoragePlacement::OwnedSlot {
                        slot: StorageSlotId::UNASSIGNED,
                    },
                    StorageLaneKind::Hip => StoragePlacement::Skipped,
                },
                RiscOp::Drop => StoragePlacement::TerminalDrop {
                    source: node.inputs[0],
                },
                RiscOp::Store { .. } => match lane {
                    StorageLaneKind::C => StoragePlacement::MaterializedStore {
                        source: node.inputs[0],
                        slot: StorageSlotId::UNASSIGNED,
                    },
                    StorageLaneKind::Hip => StoragePlacement::StoreAlias {
                        source: node.inputs[0],
                    },
                },
                RiscOp::ListMapCapture { .. }
                | RiscOp::OrderedAdjointSum { .. }
                | RiscOp::Iota
                | RiscOp::Const { .. }
                | RiscOp::ConstTensor { .. }
                | RiscOp::Shape { .. }
                | RiscOp::ExtentWitness { .. }
                | RiscOp::CheckedReshapeExtent { .. }
                | RiscOp::CheckedUnitAxis { .. }
                | RiscOp::Add
                | RiscOp::Sub
                | RiscOp::Mul
                | RiscOp::Div
                | RiscOp::FloorDiv
                | RiscOp::TruncDiv
                | RiscOp::Mod
                | RiscOp::Bitwise(_)
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
                | RiscOp::Abs
                | RiscOp::Floor
                | RiscOp::Ceil
                | RiscOp::Round
                | RiscOp::UniformLike
                | RiscOp::Dropout
                | RiscOp::DropoutReplay
                | RiscOp::UniformBoundAdjoint { .. }
                | RiscOp::Copy
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
                | RiscOp::CastTrunc { .. }
                | RiscOp::FusedElem { .. }
                | RiscOp::Pad { .. }
                | RiscOp::Shrink { .. }
                | RiscOp::BlasMatmul { .. }
                | RiscOp::Gather { .. }
                | RiscOp::ScatterAdd { .. }
                | RiscOp::Scatter { .. }
                | RiscOp::ScatterElements { .. } => StoragePlacement::OwnedSlot {
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
    dag: VerifiedDagView<'_>,
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
    dag: VerifiedDagView<'_>,
    placements: &[StoragePlacement],
    owner_of: &[Option<NodeId>],
    requirements: &mut BTreeMap<NodeId, OwnerRequirement>,
) {
    for node in dag.nodes() {
        if matches!(placements[node.id.0], StoragePlacement::Skipped) {
            continue;
        }
        let mut effective_inputs = match &node.op {
            RiscOp::Sum { .. } | RiscOp::MaxReduce { .. } => {
                if let Some(input) = node.inputs.first().copied()
                    && matches!(placements[input.0], StoragePlacement::Skipped)
                {
                    dag.get(input)
                        .expect("verified skipped reduction input names a node")
                        .inputs
                        .clone()
                } else {
                    node.inputs.clone()
                }
            }
            RiscOp::Drop => match placements[node.id.0] {
                StoragePlacement::TerminalDrop { source } => vec![source],
                _ => node.inputs.clone(),
            },
            _ => node.inputs.clone(),
        };
        // Ownership borrows shape dependencies as well as value inputs. A
        // result-claim witness stores the canonical scalar the producer reads;
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
    }
    let epilogue = dag.len();
    for root in dag.roots() {
        if let Some(owner) = owner_of[root.0]
            && let Some(requirement) = requirements.get_mut(&owner)
        {
            requirement.last_use_index = epilogue;
        }
    }
}

fn assign_slots(
    dag: VerifiedDagView<'_>,
    placements: &mut [StoragePlacement],
    owner_of: &[Option<NodeId>],
    requirements: &BTreeMap<NodeId, OwnerRequirement>,
    destructively_dropped: &BTreeSet<NodeId>,
) -> Result<AssignedSlots, OwnershipError> {
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
            let token = ReusableOwnedStorage::mint(
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
            if reusable.insert(requirement.owner, token).is_some() {
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

fn allocation_elements_for_node(dag: VerifiedDagView<'_>, node: NodeId) -> DimExpr {
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
    dag: VerifiedDagView<'_>,
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
    dag: VerifiedDagView<'_>,
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

fn physical_live_byte_bound(slots: &[StorageSlotPlan]) -> Result<LiveByteBound, OwnershipError> {
    // Logical death permits reuse; it does not release physical storage. HIP
    // retains its entire slot pool through cleanup, and C retains descriptors
    // unless explicitly dropped. Count every distinct slot once, including dead
    // gaps. This is a concrete upper bound, conservative for early C Drops, not
    // a claim that every lane necessarily observes the same exact peak.
    let mut total = 0u64;
    for slot in slots {
        let Some(bytes) = slot.capacity.allocation_bytes else {
            return Ok(LiveByteBound::Unknown);
        };
        total = total
            .checked_add(bytes)
            .ok_or_else(|| OwnershipError::LiveByteBoundOverflow {
                context: "summing distinct physical DAG slots".to_string(),
            })?;
    }
    Ok(LiveByteBound::Exact(total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{Dag, FusedInput, FusedStep, FusedStepOp, TensorType};
    use crate::ownership::{lower_dag_ownership, verify_ownership};
    use chelis_types::types::Prim;

    fn synthetic_candidate(provenance: StorageProvenance) -> ReuseCandidateFacts {
        ReuseCandidateFacts::complete_for_test(provenance)
    }

    #[test]
    fn complete_runtime_owned_proof_mints_one_linear_token() {
        let token = ReusableOwnedStorage::mint_for_test(
            NodeId(3),
            StorageId::from_index_for_test(7),
            NodeId(5),
            synthetic_candidate(StorageProvenance::RuntimeOwned),
        )
        .expect("all six conditions should mint the synthetic token");
        assert_eq!(token.source(), NodeId(3));
        assert_eq!(token.storage_for_test(), StorageId::from_index_for_test(7));
        assert_eq!(token.consumer(), NodeId(5));
    }

    #[test]
    fn every_missing_proof_condition_rejects() {
        for condition in ReuseProofCondition::ALL {
            let facts = synthetic_candidate(StorageProvenance::RuntimeOwned)
                .without_condition_for_test(condition);
            assert_eq!(
                ReusableOwnedStorage::mint_for_test(
                    NodeId(3),
                    StorageId::from_index_for_test(7),
                    NodeId(5),
                    facts,
                ),
                Err(ReuseProofRejection::Missing(condition)),
                "removing {} must reject reuse",
                condition.name()
            );
        }
    }

    #[test]
    fn entry_borrow_and_shared_view_cannot_mint_reusable_storage() {
        for provenance in [
            StorageProvenance::EntryBorrow,
            StorageProvenance::SharedView,
        ] {
            assert_eq!(
                ReusableOwnedStorage::mint_for_test(
                    NodeId(3),
                    StorageId::from_index_for_test(7),
                    NodeId(5),
                    synthetic_candidate(provenance),
                ),
                Err(ReuseProofRejection::Missing(
                    ReuseProofCondition::RuntimeOwnedAndWritable
                ))
            );
        }
    }

    #[test]
    fn hip_reshape_materializes_independent_storage_while_permute_retains_source() {
        let mut dag = crate::dag::Dag::new();
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![crate::dag::DimInfo::Lit(2), crate::dag::DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let input = dag.add_node(decl, RiscOp::Load { name: "x".into() }, vec![], ty, None);
        let permute = dag.add_node(
            decl,
            RiscOp::Permute { axes: vec![1, 0] },
            vec![input],
            TensorType {
                dims: vec![crate::dag::DimInfo::Lit(3), crate::dag::DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        let reshape = dag.add_node(
            decl,
            RiscOp::Reshape {
                new_shape: vec![crate::dag::RtDim::Lit(6)],
            },
            vec![permute],
            vector(6, Prim::F32),
            None,
        );
        dag.add_root(reshape);
        let plan = plan_hip_storage(verified(dag)).unwrap();
        assert!(
            matches!(plan.placements()[permute.0], StoragePlacement::SharedView { source } if source == input)
        );
        assert!(matches!(
            plan.placements()[reshape.0],
            StoragePlacement::OwnedSlot { .. }
        ));
        assert_ne!(
            plan.placements()[input.0].slot(),
            plan.placements()[reshape.0].slot()
        );
        assert_eq!(plan.max_live_bytes(), LiveByteBound::Exact(48));
    }

    fn vector(elements: usize, precision: Prim) -> TensorType {
        TensorType {
            dims: vec![crate::dag::DimInfo::Lit(elements)],
            precision,
        }
    }

    fn matrix(rows: usize, columns: usize, precision: Prim) -> TensorType {
        TensorType {
            dims: vec![
                crate::dag::DimInfo::Lit(rows),
                crate::dag::DimInfo::Lit(columns),
            ],
            precision,
        }
    }

    fn verified(dag: Dag) -> VerifiedDagProgram {
        verify_ownership(lower_dag_ownership(dag).expect("test DAG lowers"))
            .expect("test DAG verifies")
    }

    fn dropped_then_root(first: TensorType, second: TensorType) -> Dag {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let first_node = dag.add_node(
            decl,
            RiscOp::synth_const(first.precision, 1.0),
            vec![],
            first.clone(),
            None,
        );
        dag.add_node(decl, RiscOp::Drop, vec![first_node], first, None);
        let second_node = dag.add_node(
            decl,
            RiscOp::synth_const(second.precision, 2.0),
            vec![],
            second,
            None,
        );
        dag.add_root(second_node);
        dag
    }

    #[test]
    fn destructive_drop_retires_an_exact_capacity_slot_in_both_lanes() {
        let dag = dropped_then_root(matrix(2, 2, Prim::F32), vector(4, Prim::F32));
        let c = plan_c_storage(verified(dag.clone())).expect("same-byte C capacity plan");
        let hip = plan_hip_storage(verified(dag)).expect("same-byte HIP capacity plan");

        assert_eq!(c.slots().len(), 2);
        assert_eq!(hip.slots().len(), 2);
    }

    #[test]
    fn equal_width_different_representations_never_share_a_slot() {
        let plan = plan_c_storage(verified(dropped_then_root(
            vector(4, Prim::F32),
            vector(4, Prim::Int32),
        )))
        .expect("mixed-representation plan");
        assert_eq!(plan.slots().len(), 2);
    }

    #[test]
    fn c_excludes_entry_borrows_while_hip_counts_input_mirrors() {
        fn input_dag() -> Dag {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let input = dag.add_node(
                decl,
                RiscOp::Load { name: "x".into() },
                vec![],
                vector(4, Prim::F32),
                None,
            );
            dag.add_root(input);
            dag
        }
        let c = plan_c_storage(verified(input_dag())).expect("C storage plan");
        assert_eq!(c.max_live_bytes(), LiveByteBound::Exact(0));
        let hip = plan_hip_storage(verified(input_dag())).expect("HIP storage plan");
        assert_eq!(hip.max_live_bytes(), LiveByteBound::Exact(16));
    }

    #[test]
    fn c_materialized_store_has_storage_independent_of_its_source() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let source = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            vector(4, Prim::F32),
            None,
        );
        let store = dag.add_node(
            decl,
            RiscOp::Store { name: "out".into() },
            vec![source],
            vector(4, Prim::F32),
            None,
        );
        dag.add_root(store);
        let plan = plan_c_storage(verified(dag)).expect("C store plan");
        let StoragePlacement::MaterializedStore {
            source: actual,
            slot,
        } = plan.placement(store).expect("store placement")
        else {
            panic!("store must materialize independent storage")
        };
        assert_eq!(*actual, source);
        assert_ne!(*slot, plan.slot_for_node(source).expect("source slot"));
        assert_eq!(plan.max_live_bytes(), LiveByteBound::Exact(32));
    }

    #[test]
    fn program_owned_fused_reuse_mints_one_take_only_token() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vector(4, Prim::F32),
            None,
        );
        let owned = dag.add_node(decl, RiscOp::Copy, vec![input], vector(4, Prim::F32), None);
        let scale = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 2.0),
            vec![],
            vector(4, Prim::F32),
            None,
        );
        let fused = dag.add_node(
            decl,
            RiscOp::FusedElem {
                ops: vec![FusedStep {
                    op: FusedStepOp::Mul,
                    input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
                }],
            },
            vec![owned, scale],
            vector(4, Prim::F32),
            None,
        );
        dag.set_reusable_input(fused, owned);
        dag.add_root(fused);

        let mut plan = plan_c_storage(verified(dag)).expect("C reuse plan");
        let token = plan
            .take_reuse_for(fused)
            .expect("first take succeeds")
            .expect("owned fused input is reusable");
        assert_eq!(token.source(), owned);
        assert_eq!(token.consumer(), fused);
        assert!(plan.take_reuse_for(fused).is_err());
    }
}
