//! C emission adapter for the shared `chelis-ir` storage plan.

use chelis_ir::dag::{DimExpr, NodeId};
use chelis_ir::ownership::{
    CStorageLane, StoragePlacement, StorageSlotPlan, VerifiedDagView, VerifiedStorageLayout,
    VerifiedStoragePlan,
};
use chelis_types::types::Prim;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeMemoryKind {
    BorrowedLoad,
    SlotBacked { slot: usize },
    MetadataView { source: NodeId },
    TerminalDrop { source: NodeId },
    StandaloneStore { source: NodeId },
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotPlan {
    pub id: usize,
    pub dtype: Prim,
    pub capacity_elems: DimExpr,
    pub first_owner: NodeId,
    pub last_use_index: usize,
}

#[derive(Debug)]
pub struct MemoryPlan {
    node_kinds: Vec<NodeMemoryKind>,
    slots: Vec<SlotPlan>,
}

impl MemoryPlan {
    pub fn from_shared(plan: &VerifiedStoragePlan<CStorageLane>) -> Self {
        Self::from_parts(plan.emission(), plan.placements(), plan.slots())
    }

    pub fn from_layout(plan: &VerifiedStorageLayout<'_, CStorageLane>) -> Self {
        Self::from_parts(plan.emission(), plan.placements(), plan.slots())
    }

    fn from_parts(
        dag: VerifiedDagView<'_>,
        placements: &[StoragePlacement],
        shared_slots: &[StorageSlotPlan],
    ) -> Self {
        let node_kinds = placements
            .iter()
            .map(|placement| match placement {
                StoragePlacement::BorrowedEntry => NodeMemoryKind::BorrowedLoad,
                StoragePlacement::OwnedSlot { slot } => {
                    NodeMemoryKind::SlotBacked { slot: slot.index() }
                }
                StoragePlacement::SharedView { source } => {
                    NodeMemoryKind::MetadataView { source: *source }
                }
                StoragePlacement::TerminalDrop { source } => {
                    NodeMemoryKind::TerminalDrop { source: *source }
                }
                StoragePlacement::MaterializedStore { source, .. } => {
                    NodeMemoryKind::StandaloneStore { source: *source }
                }
                StoragePlacement::Skipped => NodeMemoryKind::Skipped,
                StoragePlacement::InputMirror { .. } | StoragePlacement::StoreAlias { .. } => {
                    unreachable!("HIP placement reached the C adapter")
                }
            })
            .collect();
        let slots = shared_slots
            .iter()
            .map(|slot| SlotPlan {
                id: slot.id().index(),
                dtype: dag
                    .get(slot.first_owner())
                    .expect("shared slot owner belongs to the verified DAG")
                    .output_type
                    .precision,
                capacity_elems: slot.allocation_elements().clone(),
                first_owner: slot.first_owner(),
                last_use_index: slot.last_use_index(),
            })
            .collect();
        Self { node_kinds, slots }
    }

    pub fn node_kind(&self, id: NodeId) -> &NodeMemoryKind {
        &self.node_kinds[id.0]
    }

    pub fn slot(&self, id: usize) -> &SlotPlan {
        &self.slots[id]
    }

    pub fn slots(&self) -> &[SlotPlan] {
        &self.slots
    }

    pub fn emit_cleanup(&self, output_ids: &[NodeId]) -> Vec<String> {
        self.emit_cleanup_with_drops(output_ids, &[])
    }

    pub fn emit_cleanup_with_drops(
        &self,
        output_ids: &[NodeId],
        dropped_sources: &[NodeId],
    ) -> Vec<String> {
        let mut lines = Vec::new();
        for (idx, kind) in self.node_kinds.iter().enumerate() {
            let id = NodeId(idx);
            if output_ids.contains(&id)
                || dropped_sources.contains(&id)
                || matches!(
                    kind,
                    NodeMemoryKind::BorrowedLoad
                        | NodeMemoryKind::TerminalDrop { .. }
                        | NodeMemoryKind::Skipped
                )
            {
                continue;
            }
            lines.push(format!("    chelis_tensor_release(t{idx});"));
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
    use chelis_ir::ownership::{lower_dag_ownership, plan_c_storage, verify_ownership};

    fn vec_f32(elements: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(elements)],
            precision: Prim::F32,
        }
    }

    fn build(dag: Dag) -> MemoryPlan {
        let verified = verify_ownership(lower_dag_ownership(dag).unwrap()).unwrap();
        let shared = plan_c_storage(verified).unwrap();
        MemoryPlan::from_shared(&shared)
    }

    #[test]
    fn adapter_preserves_borrowed_and_exact_slot_placements() {
        let mut dag = Dag::new();
        let input = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let output = dag.add_node(RiscOp::Neg, vec![input], vec_f32(4), None);
        dag.add_root(output);
        let plan = build(dag);
        assert_eq!(plan.node_kind(input), &NodeMemoryKind::BorrowedLoad);
        assert_eq!(
            plan.node_kind(output),
            &NodeMemoryKind::SlotBacked { slot: 0 }
        );
    }

    #[test]
    fn adapter_has_no_capacity_comparison_or_eligibility_api() {
        let source = include_str!("memory.rs");
        for forbidden in [
            ["DimExpr", "Key"].concat(),
            ["capacity", "_fits"].concat(),
            ["borrows", "_caller_storage"].concat(),
            ["logical", "_elements"].concat(),
        ] {
            assert!(!source.contains(&forbidden), "found {forbidden}");
        }
    }
}
