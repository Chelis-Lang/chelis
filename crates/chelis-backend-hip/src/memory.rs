//! GPU memory planning for generated HIP code.
//!
//! Phase 1c adds a simple greedy slot planner:
//! - storage-owning values (unique loads and materialized producers) get a slot
//! - repeated loads alias the first unique load for that input label
//! - movement/store nodes are metadata-only views
//! - reduction-inlined fused elementwise nodes are skipped entirely
//! - cleanup frees every metadata wrapper, then each slot exactly once

use std::collections::{HashMap, HashSet};

use chelis_ir::dag::{Dag, DimExpr, DimInfo, NodeId, RiscOp};
use chelis_types::types::Prim;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeMemoryKind {
    UniqueInput { slot: usize, input_name: String },
    RepeatedLoadAlias { canonical_load: NodeId },
    SlotBacked { slot: usize },
    MetadataView { source: NodeId },
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

#[derive(Debug, Clone)]
pub struct MemoryPlan {
    node_kinds: Vec<NodeMemoryKind>,
    slots: Vec<SlotPlan>,
}

#[derive(Debug, Clone)]
struct OwnerRequirement {
    owner: NodeId,
    birth_index: usize,
    last_use_index: usize,
    capacity_elems: DimExpr,
    dtype: Prim,
}

impl MemoryPlan {
    pub fn build(dag: &Dag, output_ids: &[NodeId], reduction_inlined: &HashSet<NodeId>) -> Self {
        let mut node_kinds = classify_nodes(dag, reduction_inlined);
        let owner_of = compute_owner_map(dag, &node_kinds);
        let requirements = owner_requirements(dag, &node_kinds, &owner_of, output_ids);
        let slots = assign_slots(&requirements, &mut node_kinds);
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

    pub fn peak_device_bytes_estimate(&self) -> Option<usize> {
        self.peak_device_bytes_terms()
            .into_iter()
            .try_fold(0usize, |acc, term| Some(acc + term.as_concrete()?))
    }

    pub fn peak_device_bytes_formula(&self) -> String {
        render_dim_expr_sum(&self.peak_device_bytes_terms())
    }

    pub fn peak_device_bytes_at(&self, bindings: &HashMap<String, usize>) -> Result<usize, String> {
        self.peak_device_bytes_terms()
            .into_iter()
            .try_fold(0usize, |acc, term| Ok(acc + term.evaluate(bindings)?))
    }

    pub fn emit_cleanup(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for (idx, kind) in self.node_kinds.iter().enumerate() {
            if matches!(kind, NodeMemoryKind::Skipped) {
                continue;
            }
            lines.push(format!("    chelis_gpu_free_view(d_t{idx});"));
        }
        for slot in &self.slots {
            lines.push(format!("    chelis_gpu_free(chelis_slot{});", slot.id));
        }
        lines
    }
}

fn classify_nodes(dag: &Dag, reduction_inlined: &HashSet<NodeId>) -> Vec<NodeMemoryKind> {
    let mut kinds = Vec::with_capacity(dag.len());
    let mut first_load_by_name: HashMap<String, NodeId> = HashMap::new();

    for node in dag.nodes() {
        let kind = if reduction_inlined.contains(&node.id) {
            NodeMemoryKind::Skipped
        } else {
            match &node.op {
                RiscOp::Load { name } => {
                    if let Some(&canonical_load) = first_load_by_name.get(name) {
                        NodeMemoryKind::RepeatedLoadAlias { canonical_load }
                    } else {
                        first_load_by_name.insert(name.clone(), node.id);
                        NodeMemoryKind::UniqueInput {
                            slot: usize::MAX,
                            input_name: name.clone(),
                        }
                    }
                }
                RiscOp::Reshape { .. }
                | RiscOp::Permute { .. }
                | RiscOp::Expand { .. }
                | RiscOp::Stride { .. } => NodeMemoryKind::MetadataView {
                    source: node.inputs[0],
                },
                RiscOp::Store { .. } => NodeMemoryKind::StoreAlias {
                    source: node.inputs[0],
                },
                RiscOp::Const { .. }
                | RiscOp::Add
                | RiscOp::Mul
                | RiscOp::CmpLt
                | RiscOp::MaxElem
                | RiscOp::Neg
                | RiscOp::Exp
                | RiscOp::Log
                | RiscOp::Sin
                | RiscOp::Sqrt
                | RiscOp::Dropout { .. }
                | RiscOp::Sum { .. }
                | RiscOp::MaxReduce { .. }
                | RiscOp::Realize
                | RiscOp::Cast { .. }
                | RiscOp::FusedElem { .. }
                | RiscOp::Pad { .. }
                | RiscOp::Shrink { .. } => NodeMemoryKind::SlotBacked { slot: usize::MAX },
            }
        };
        kinds.push(kind);
    }

    kinds
}

fn compute_owner_map(dag: &Dag, node_kinds: &[NodeMemoryKind]) -> Vec<Option<NodeId>> {
    let mut owners = vec![None; dag.len()];
    for node in dag.nodes() {
        owners[node.id.0] = match &node_kinds[node.id.0] {
            NodeMemoryKind::UniqueInput { .. } | NodeMemoryKind::SlotBacked { .. } => Some(node.id),
            NodeMemoryKind::RepeatedLoadAlias { canonical_load } => owners[canonical_load.0],
            NodeMemoryKind::MetadataView { source } | NodeMemoryKind::StoreAlias { source } => {
                owners[source.0]
            }
            NodeMemoryKind::Skipped => None,
        };
    }
    owners
}

fn owner_requirements(
    dag: &Dag,
    node_kinds: &[NodeMemoryKind],
    owner_of: &[Option<NodeId>],
    output_ids: &[NodeId],
) -> Vec<OwnerRequirement> {
    let epilogue_index = dag.len();
    let mut by_owner = HashMap::<NodeId, OwnerRequirement>::new();

    for node in dag.nodes() {
        match node_kinds[node.id.0] {
            NodeMemoryKind::UniqueInput { .. } | NodeMemoryKind::SlotBacked { .. } => {
                by_owner.insert(
                    node.id,
                    OwnerRequirement {
                        owner: node.id,
                        birth_index: node.id.0,
                        last_use_index: node.id.0,
                        capacity_elems: logical_elements(&node.output_type),
                        dtype: node.output_type.precision,
                    },
                );
            }
            _ => {}
        }
    }

    for node in dag.nodes() {
        if matches!(node_kinds[node.id.0], NodeMemoryKind::Skipped) {
            continue;
        }

        let effective_inputs: Vec<NodeId> = match &node.op {
            RiscOp::Sum { .. } | RiscOp::MaxReduce { .. } => {
                if let Some(fused_input) = node.inputs.first().copied()
                    && matches!(node_kinds[fused_input.0], NodeMemoryKind::Skipped)
                {
                    dag.get(fused_input)
                        .map(|fused_node| fused_node.inputs.clone())
                        .unwrap_or_default()
                } else {
                    node.inputs.clone()
                }
            }
            _ => node.inputs.clone(),
        };

        for input in effective_inputs {
            if let Some(owner) = owner_of[input.0]
                && let Some(req) = by_owner.get_mut(&owner)
            {
                req.last_use_index = req.last_use_index.max(node.id.0);
            }
        }
    }

    for &output_id in output_ids {
        let output_node = dag.get(output_id).expect("output id must exist");
        if matches!(output_node.op, RiscOp::Load { .. }) {
            continue;
        }
        if let Some(owner) = owner_of[output_id.0]
            && let Some(req) = by_owner.get_mut(&owner)
        {
            req.last_use_index = epilogue_index;
        }
    }

    let mut ordered = by_owner.into_values().collect::<Vec<_>>();
    ordered.sort_by_key(|req| req.birth_index);
    ordered
}

fn assign_slots(
    requirements: &[OwnerRequirement],
    node_kinds: &mut [NodeMemoryKind],
) -> Vec<SlotPlan> {
    let mut slots = Vec::<SlotPlan>::new();
    let mut availability: Vec<usize> = Vec::new();
    let mut owner_to_slot = HashMap::<NodeId, usize>::new();

    for req in requirements {
        let reused = slots.iter().enumerate().find_map(|(slot_id, slot)| {
            let reusable = slot.dtype == req.dtype
                && availability[slot_id] < req.birth_index
                && match (
                    slot.capacity_elems.as_concrete(),
                    req.capacity_elems.as_concrete(),
                ) {
                    (Some(slot_elems), Some(req_elems)) => slot_elems >= req_elems,
                    _ => slot.capacity_elems == req.capacity_elems,
                };
            reusable.then_some(slot_id)
        });

        let slot_id = reused.unwrap_or_else(|| {
            let slot_id = slots.len();
            slots.push(SlotPlan {
                id: slot_id,
                dtype: req.dtype,
                capacity_elems: req.capacity_elems.clone(),
                first_owner: req.owner,
                last_use_index: req.last_use_index,
            });
            availability.push(usize::MIN);
            slot_id
        });

        availability[slot_id] = req.last_use_index;
        slots[slot_id].last_use_index = slots[slot_id].last_use_index.max(req.last_use_index);
        owner_to_slot.insert(req.owner, slot_id);
    }

    for (idx, kind) in node_kinds.iter_mut().enumerate() {
        match kind {
            NodeMemoryKind::UniqueInput { slot, .. } | NodeMemoryKind::SlotBacked { slot } => {
                let owner = NodeId(idx);
                let slot_id = owner_to_slot[&owner];
                *slot = slot_id;
            }
            _ => {}
        }
    }

    slots
}

fn logical_elements(ty: &chelis_ir::dag::TensorType) -> DimExpr {
    if ty.dims.is_empty() {
        DimExpr::Concrete(1)
    } else {
        ty.dims
            .iter()
            .map(dim_size)
            .reduce(|lhs, rhs| DimExpr::Mul(Box::new(lhs), Box::new(rhs)))
            .unwrap_or(DimExpr::Concrete(1))
    }
}

fn dim_size(dim: &DimInfo) -> DimExpr {
    match dim {
        DimInfo::Lit(n) => DimExpr::Concrete(*n),
        DimInfo::Named(_, Some(n)) => DimExpr::Concrete(*n),
        DimInfo::Named(name, None) => DimExpr::Sym(name.clone()),
    }
}

fn bytes_expr(expr: &DimExpr, dtype: Prim) -> DimExpr {
    let bytes = bytes_per_element(dtype);
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
            .map(|expr| expr.to_string())
            .collect::<Vec<_>>()
            .join(" + ")
    }
}

impl MemoryPlan {
    pub(crate) fn peak_device_bytes_terms(&self) -> Vec<DimExpr> {
        self.slots
            .iter()
            .map(|slot| bytes_expr(&slot.capacity_elems, slot.dtype))
            .collect()
    }
}

fn bytes_per_element(dtype: Prim) -> usize {
    match dtype {
        Prim::F32 | Prim::Bool => 4,
        other => panic!(
            "Phase 1c HIP memory planner only supports f32/bool tensors, got {}",
            other.name()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_ir::dag::{Dag, DimInfo, TensorType};

    fn scalar_f32() -> chelis_ir::dag::TensorType {
        chelis_ir::dag::TensorType::scalar_f32()
    }

    fn vec_f32(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: Prim::F32,
        }
    }

    fn build_plan(dag: &Dag, output_ids: &[NodeId]) -> MemoryPlan {
        MemoryPlan::build(dag, output_ids, &HashSet::new())
    }

    #[test]
    fn planner_reuses_non_overlapping_slots() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
        let d = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let e = dag.add_node(RiscOp::Mul, vec![c, d], scalar_f32());
        dag.add_root(e);

        let plan = build_plan(&dag, &[e]);

        assert_eq!(plan.slots().len(), 3);
        let d_slot = match plan.node_kind(d) {
            NodeMemoryKind::SlotBacked { slot } => *slot,
            other => panic!("expected slot-backed const, got {other:?}"),
        };
        let a_slot = match plan.node_kind(a) {
            NodeMemoryKind::SlotBacked { slot } => *slot,
            other => panic!("expected slot-backed const, got {other:?}"),
        };
        assert_eq!(d_slot, a_slot, "later const should reuse the dead slot");
    }

    #[test]
    fn planner_keeps_overlapping_values_separate() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4));
        let y = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], vec_f32(4));
        let add = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(4));
        let mul = dag.add_node(RiscOp::Mul, vec![x, y], vec_f32(4));
        dag.add_root(add);
        dag.add_root(mul);

        let plan = build_plan(&dag, &[add, mul]);

        let x_slot = match plan.node_kind(x) {
            NodeMemoryKind::SlotBacked { slot } => *slot,
            other => panic!("expected slot-backed const, got {other:?}"),
        };
        let y_slot = match plan.node_kind(y) {
            NodeMemoryKind::SlotBacked { slot } => *slot,
            other => panic!("expected slot-backed const, got {other:?}"),
        };
        assert_ne!(x_slot, y_slot, "live inputs must not alias");
    }

    #[test]
    fn planner_dedups_repeated_loads() {
        let mut dag = Dag::new();
        let x0 = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4));
        let x1 = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4));
        let add = dag.add_node(RiscOp::Add, vec![x0, x1], vec_f32(4));
        dag.add_root(add);

        let plan = build_plan(&dag, &[add]);

        match plan.node_kind(x0) {
            NodeMemoryKind::UniqueInput { .. } => {}
            other => panic!("first load should own the device copy, got {other:?}"),
        }
        match plan.node_kind(x1) {
            NodeMemoryKind::RepeatedLoadAlias { canonical_load } => {
                assert_eq!(*canonical_load, x0);
            }
            other => panic!("second load should alias the first, got {other:?}"),
        }
        assert_eq!(
            plan.slots().len(),
            2,
            "one input slot plus one add output slot"
        );
    }

    #[test]
    fn cleanup_frees_wrappers_then_slots() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        dag.add_root(a);

        let plan = build_plan(&dag, &[a]);
        let lines = plan.emit_cleanup();

        assert_eq!(lines[0], "    chelis_gpu_free_view(d_t0);");
        assert_eq!(lines[1], "    chelis_gpu_free(chelis_slot0);");
    }

    #[test]
    fn planner_reuses_identical_symbolic_slots() {
        let mut dag = Dag::new();
        let symbolic = TensorType {
            dims: vec![DimInfo::Named("batch".into(), None)],
            precision: Prim::F32,
        };
        let a = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], symbolic.clone());
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], symbolic.clone());
        let add = dag.add_node(RiscOp::Add, vec![a, b], symbolic.clone());
        let two = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], symbolic.clone());
        let neg = dag.add_node(RiscOp::Neg, vec![two], symbolic.clone());
        dag.add_root(neg);

        let plan = build_plan(&dag, &[neg]);
        assert_eq!(
            plan.slots().len(),
            3,
            "identical symbolic capacities should reuse dead slots instead of allocating one slot per owner"
        );
        assert!(matches!(
            plan.node_kind(add),
            NodeMemoryKind::SlotBacked { .. }
        ));
        assert!(matches!(
            plan.node_kind(neg),
            NodeMemoryKind::SlotBacked { .. }
        ));
    }

    #[test]
    fn symbolic_peak_device_memory_reports_formula_and_eval() {
        let mut dag = Dag::new();
        let symbolic = TensorType {
            dims: vec![DimInfo::Named("batch".into(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], symbolic.clone());
        dag.add_root(x);

        let plan = build_plan(&dag, &[x]);
        assert_eq!(plan.peak_device_bytes_estimate(), None);
        assert_eq!(plan.peak_device_bytes_formula(), "(batch * 4)");

        let bindings = HashMap::from([(String::from("batch"), 32usize)]);
        assert_eq!(plan.peak_device_bytes_at(&bindings).unwrap(), 128);
    }
}
