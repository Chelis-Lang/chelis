//! HIP emission adapter for the shared `chelis-ir` storage plan.

use chelis_unord::UnordMap;

use chelis_ir::dag::{DimExpr, NodeId, RiscOp};
use chelis_ir::ownership::{HipStorageLane, LiveByteBound, StoragePlacement, VerifiedStoragePlan};
use chelis_types::types::Prim;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeMemoryKind {
    UniqueInput { slot: usize, input_name: String },
    RepeatedLoadAlias { canonical_load: NodeId },
    SlotBacked { slot: usize },
    MetadataView { source: NodeId },
    TerminalDrop { source: NodeId },
    StoreAlias { source: NodeId },
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
    max_live_bytes: LiveByteBound,
}

impl MemoryPlan {
    pub fn from_shared(plan: &VerifiedStoragePlan<HipStorageLane>) -> Self {
        let dag = plan.emission();
        let node_kinds = plan
            .placements()
            .iter()
            .enumerate()
            .map(|(index, placement)| match placement {
                StoragePlacement::InputMirror { slot, input_name } => NodeMemoryKind::UniqueInput {
                    slot: slot.index(),
                    input_name: input_name.clone(),
                },
                StoragePlacement::OwnedSlot { slot } => {
                    NodeMemoryKind::SlotBacked { slot: slot.index() }
                }
                StoragePlacement::SharedView { source }
                    if matches!(
                        dag.get(NodeId(index)).map(|node| &node.op),
                        Some(RiscOp::Load { .. })
                    ) =>
                {
                    NodeMemoryKind::RepeatedLoadAlias {
                        canonical_load: *source,
                    }
                }
                StoragePlacement::SharedView { source } => {
                    NodeMemoryKind::MetadataView { source: *source }
                }
                StoragePlacement::TerminalDrop { source } => {
                    NodeMemoryKind::TerminalDrop { source: *source }
                }
                StoragePlacement::StoreAlias { source } => {
                    NodeMemoryKind::StoreAlias { source: *source }
                }
                StoragePlacement::Skipped => NodeMemoryKind::Skipped,
                StoragePlacement::BorrowedEntry | StoragePlacement::MaterializedStore { .. } => {
                    unreachable!("C placement reached the HIP adapter")
                }
            })
            .collect();
        let slots = plan
            .slots()
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
        Self {
            node_kinds,
            slots,
            max_live_bytes: plan.max_live_bytes(),
        }
    }

    pub fn node_kind(&self, id: NodeId) -> &NodeMemoryKind {
        &self.node_kinds[id.0]
    }

    pub fn iter_node_kinds(&self) -> impl Iterator<Item = &NodeMemoryKind> {
        self.node_kinds.iter()
    }

    pub fn slot(&self, id: usize) -> &SlotPlan {
        &self.slots[id]
    }

    pub fn slots(&self) -> &[SlotPlan] {
        &self.slots
    }

    pub(crate) fn slot_has_later_owner(&self, id: NodeId) -> bool {
        let slot_id = match self.node_kind(id) {
            NodeMemoryKind::UniqueInput { slot, .. } | NodeMemoryKind::SlotBacked { slot } => *slot,
            other => panic!("node {} does not own slot-backed storage: {other:?}", id.0),
        };
        self.node_kinds.iter().skip(id.0 + 1).any(|kind| {
            matches!(
                kind,
                NodeMemoryKind::UniqueInput { slot, .. } | NodeMemoryKind::SlotBacked { slot }
                    if *slot == slot_id
            )
        })
    }

    pub fn peak_device_bytes_estimate(&self) -> Option<usize> {
        match self.max_live_bytes {
            LiveByteBound::Exact(bytes) => usize::try_from(bytes).ok(),
            LiveByteBound::Unknown | LiveByteBound::Unbounded => None,
        }
    }

    pub fn peak_device_bytes_formula(&self) -> String {
        render_dim_expr_sum(&self.peak_device_bytes_terms())
    }

    pub fn peak_device_bytes_at(
        &self,
        bindings: &UnordMap<String, usize>,
    ) -> Result<usize, String> {
        self.peak_device_bytes_terms()
            .into_iter()
            .try_fold(0usize, |acc, term| Ok(acc + term.evaluate(bindings)?))
    }

    pub(crate) fn peak_device_bytes_terms(&self) -> Vec<DimExpr> {
        self.slots
            .iter()
            .map(|slot| bytes_expr(&slot.capacity_elems, slot.dtype))
            .collect()
    }

    pub fn emit_cleanup(&self) -> Vec<String> {
        self.emit_cleanup_with_drops(&[])
    }

    pub fn emit_cleanup_with_drops(&self, dropped_sources: &[NodeId]) -> Vec<String> {
        let mut lines = Vec::new();
        for (idx, kind) in self.node_kinds.iter().enumerate() {
            if dropped_sources.contains(&NodeId(idx))
                || matches!(
                    kind,
                    NodeMemoryKind::TerminalDrop { .. } | NodeMemoryKind::Skipped
                )
            {
                continue;
            }
            lines.push(format!("    chelis_device_tensor_release(o_t{idx});"));
        }
        for slot in &self.slots {
            lines.push(format!(
                "    if (chelis_slot{0}) chelis_device_tensor_release(chelis_slot{0});",
                slot.id
            ));
        }
        lines
    }
}

pub(crate) fn bytes_expr(expr: &DimExpr, dtype: Prim) -> DimExpr {
    let bytes = dtype
        .runtime_dtype()
        .unwrap_or_else(|error| {
            panic!(
                "HIP storage adapter cannot size `{}`: {error}",
                dtype.name()
            )
        })
        .byte_width();
    if bytes == 1 {
        expr.clone()
    } else {
        DimExpr::Mul(Box::new(expr.clone()), Box::new(DimExpr::Concrete(bytes)))
    }
}

fn render_dim_expr_sum(terms: &[DimExpr]) -> String {
    if terms.is_empty() {
        "0".to_string()
    } else {
        terms
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" + ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_ir::dag::{Dag, DimInfo, TensorType};
    use chelis_ir::ownership::{lower_dag_ownership, plan_hip_storage, verify_ownership};

    fn vec_f32(elements: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(elements)],
            precision: Prim::F32,
        }
    }

    #[test]
    fn adapter_preserves_input_mirror_and_owned_slot_placements() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(4),
            None,
        );
        let output = dag.add_node(decl, RiscOp::Neg, vec![input], vec_f32(4), None);
        dag.add_root(output);
        let verified = verify_ownership(lower_dag_ownership(dag).unwrap()).unwrap();
        let shared = plan_hip_storage(verified).unwrap();
        let plan = MemoryPlan::from_shared(&shared);
        assert!(matches!(
            plan.node_kind(input),
            NodeMemoryKind::UniqueInput { .. }
        ));
        assert!(matches!(
            plan.node_kind(output),
            NodeMemoryKind::SlotBacked { .. }
        ));
    }

    #[test]
    fn adapter_has_no_capacity_comparison_or_eligibility_api() {
        let source = include_str!("memory.rs");
        for forbidden in [
            ["DimExpr", "Key"].concat(),
            ["capacity", "_fits"].concat(),
            ["logical", "_elements"].concat(),
        ] {
            assert!(!source.contains(&forbidden), "found {forbidden}");
        }
    }
}
