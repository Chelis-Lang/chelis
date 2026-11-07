//! Shared exact-capacity storage planning for the C and HIP lanes.

use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;

use chelis_dag_core::dag::{NodeId, RiscOp};

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

#[cfg(test)]
use chelis_ir_core::ownership::storage::{
    ReuseCandidateFacts, ReuseProofCondition, ReuseProofRejection, StorageId, StorageProvenance,
};
use chelis_ir_core::ownership::storage::{
    ReuseDecision, StoragePlacement, StoragePolicy, StorageSlotId, StorageSlotPlan,
    build_storage_plan as calculate_storage_plan,
};

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
/// # use chelis_dag_core::dag::NodeId;
/// use chelis_ir::ownership::ReusableOwnedStorage;
/// fn forge(source: NodeId, consumer: NodeId) -> ReusableOwnedStorage {
///     ReusableOwnedStorage { source, storage: todo!(), consumer }
/// }
/// ```
#[derive(Debug, Eq, PartialEq)]
pub struct ReusableOwnedStorage {
    data: ReuseDecision,
}

impl ReusableOwnedStorage {
    pub const fn source(&self) -> NodeId {
        self.data.source
    }

    pub const fn consumer(&self) -> NodeId {
        self.data.consumer
    }

    #[cfg(test)]
    fn mint(
        source: NodeId,
        storage: StorageId,
        consumer: NodeId,
        facts: ReuseCandidateFacts,
    ) -> Result<Self, ReuseProofRejection> {
        ReuseDecision::calculate(source, storage, consumer, facts).map(|data| Self { data })
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
        self.data.storage
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

/// Borrowing form used when a verified DAG is owned by a verified host program.
/// It carries the exact parent-owned DAG view and the same linear token set.
/// standalone backend entry points use [`VerifiedStoragePlan`].
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

fn build_storage_plan(
    dag: VerifiedDagView<'_>,
    lane: StorageLaneKind,
) -> Result<
    chelis_ir_core::ownership::storage::BuiltStoragePlan<ReusableOwnedStorage>,
    OwnershipError,
> {
    let views = (lane == StorageLaneKind::C).then(|| dag.contraction_views());
    let mut skipped = dag.reduction_inlined_fused_elems();
    if let Some(views) = &views {
        skipped.extend(
            dag.nodes()
                .iter()
                .filter(|node| views.is_deferred(node.id))
                .map(|node| node.id),
        );
    }
    match lane {
        StorageLaneKind::C => skipped.extend(
            dag.reduction_inlined_products()
                .iter()
                .flat_map(chelis_dag_core::fuse::InlinedProduct::skipped),
        ),
        StorageLaneKind::Hip => skipped.extend(hip_emission_literals(dag)),
    }
    let policy = match lane {
        StorageLaneKind::C => StoragePolicy {
            mirror_inputs: false,
            shared_views: false,
            skipped_keys: false,
            alias_stores: false,
            recurrence_tape: true,
        },
        StorageLaneKind::Hip => StoragePolicy {
            mirror_inputs: true,
            shared_views: true,
            skipped_keys: true,
            alias_stores: true,
            recurrence_tape: false,
        },
    };
    calculate_storage_plan(dag.analysis(), policy, &skipped, views.as_ref(), |data| {
        ReusableOwnedStorage { data }
    })
}

/// The constants the HIP emitter reads only as literals while it emits a draw.
/// A draw key's seed and controls, a key derivation's seed or index.
/// And a `uniform_like`'s bounds, which the emitter folds into the key it computes and the kernel's arguments.
/// Such a constant has no device value, so it takes no slot and no lifetime.
/// A constant any other operation, dependency or root reads keeps its storage.
///
fn hip_emission_literals(dag: VerifiedDagView<'_>) -> Vec<NodeId> {
    // `None` until a read is seen.
    // then whether every read is a literal one.
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ownership::{lower_dag_ownership, verify_ownership};
    use chelis_dag_core::dag::{
        Dag, FusedInput, FusedStep, FusedStepOp, RuntimeRecurrence, RuntimeRecurrenceSource,
        TensorType,
    };
    use chelis_numeric_core::prim::Prim;

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
        let mut dag = chelis_dag_core::dag::Dag::new();
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![
                chelis_dag_core::dag::DimInfo::Lit(2),
                chelis_dag_core::dag::DimInfo::Lit(3),
            ],
            precision: Prim::F32,
        };
        let input = dag.add_node(decl, RiscOp::Load { name: "x".into() }, vec![], ty, None);
        let permute = dag.add_node(
            decl,
            RiscOp::Permute { axes: vec![1, 0] },
            vec![input],
            TensorType {
                dims: vec![
                    chelis_dag_core::dag::DimInfo::Lit(3),
                    chelis_dag_core::dag::DimInfo::Lit(2),
                ],
                precision: Prim::F32,
            },
            None,
        );
        let reshape = dag.add_node(
            decl,
            RiscOp::Reshape {
                new_shape: vec![chelis_dag_core::dag::RtDim::Lit(6)],
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
            dims: vec![chelis_dag_core::dag::DimInfo::Lit(elements)],
            precision,
        }
    }

    fn scalar(precision: Prim) -> TensorType {
        TensorType {
            dims: vec![],
            precision,
        }
    }

    fn verified(dag: Dag) -> VerifiedDagProgram {
        verify_ownership(lower_dag_ownership(dag).expect("test DAG lowers"))
            .expect("test DAG verifies")
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

    #[test]
    fn c_recurrence_tape_pins_seed_capture_and_fold_result_but_not_source() {
        let mut callback = Dag::new();
        let callback_decl = callback.declare("callback");
        let state = callback.add_node(
            callback_decl,
            RiscOp::Load {
                name: "state".into(),
            },
            vec![],
            scalar(Prim::F32),
            None,
        );
        let item = callback.add_node(
            callback_decl,
            RiscOp::Load {
                name: "item".into(),
            },
            vec![],
            scalar(Prim::Int64),
            None,
        );
        let captured = callback.add_node(
            callback_decl,
            RiscOp::Load {
                name: "captured".into(),
            },
            vec![],
            scalar(Prim::F32),
            None,
        );
        let item_float = callback.add_node(
            callback_decl,
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            vec![item],
            scalar(Prim::F32),
            None,
        );
        let sum = callback.add_node(
            callback_decl,
            RiscOp::Add,
            vec![state, captured],
            scalar(Prim::F32),
            None,
        );
        let result = callback.add_node(
            callback_decl,
            RiscOp::Add,
            vec![sum, item_float],
            scalar(Prim::F32),
            None,
        );
        callback.add_root(result);

        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let source = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::Int64, 1.0),
            vec![],
            vector(1, Prim::Int64),
            None,
        );
        let seed = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 0.0),
            vec![],
            scalar(Prim::F32),
            None,
        );
        let capture = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            scalar(Prim::F32),
            None,
        );
        let fold = dag.add_node(
            decl,
            RiscOp::RuntimeRecurrence(Box::new(RuntimeRecurrence {
                callback: Box::new(callback),
                state,
                source: RuntimeRecurrenceSource::Tensor { item },
                captures: vec![("captured".into(), captured)],
                result,
                scan: false,
            })),
            vec![source, seed, capture],
            scalar(Prim::F32),
            None,
        );
        let after_f32_a = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 2.0),
            vec![],
            scalar(Prim::F32),
            None,
        );
        let after_f32_b = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 3.0),
            vec![],
            scalar(Prim::F32),
            None,
        );
        let after_f32_c = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 4.0),
            vec![],
            scalar(Prim::F32),
            None,
        );
        let after_i64 = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::Int64, 2.0),
            vec![],
            vector(1, Prim::Int64),
            None,
        );
        for root in [after_f32_a, after_f32_b, after_f32_c, after_i64] {
            dag.add_root(root);
        }
        let plan = plan_c_storage(verified(dag)).expect("C recurrence storage plan");
        for pinned in [seed, capture, fold] {
            for after in [after_f32_a, after_f32_b, after_f32_c] {
                assert_ne!(
                    plan.slot_for_node(pinned),
                    plan.slot_for_node(after),
                    "taped owner {pinned:?} must not be reused by {after:?}"
                );
            }
        }
        assert_eq!(
            plan.slot_for_node(source),
            plan.slot_for_node(after_i64),
            "the source is gathered per row and need not remain pinned"
        );
    }
}
