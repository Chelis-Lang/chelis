//! Memory planning for generated C code.
//!
//! M2a adds conservative slot planning to the C backend:
//! - caller loads are borrowed and never freed here, and (chelis#933)
//!   never written through: the buffer belongs to whoever called the
//!   compiled entry point, so an optimization may read it but must not
//!   use it as a destination
//! - materialized nodes get reusable backing slots plus per-node metadata views
//! - movement nodes are metadata views over their source owner
//! - stores remain standalone owned outputs
//! - cleanup frees metadata wrappers before backing slots

use std::collections::{HashMap, HashSet};

use chelis_ir::dag::{Dag, DimExpr, DimExprKey, DimInfo, NodeId, RiscOp, TensorType};
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
    capacity_concrete: Option<usize>,
    capacity_key: DimExprKey,
    dtype: Prim,
}

impl MemoryPlan {
    pub fn build(dag: &Dag, output_ids: &[NodeId], skipped: &HashSet<NodeId>) -> Self {
        let mut node_kinds = classify_nodes(dag, skipped);
        let owner_of = compute_owner_map(dag, &node_kinds);
        let requirements = owner_requirements(dag, &node_kinds, &owner_of, output_ids);
        let slots = assign_slots(&requirements, &mut node_kinds);
        Self { node_kinds, slots }
    }

    pub fn node_kind(&self, id: NodeId) -> &NodeMemoryKind {
        &self.node_kinds[id.0]
    }

    /// chelis#933: true when `id`'s bytes live in a buffer the caller
    /// supplied, rather than in a slot this program allocated.
    ///
    /// `BorrowedLoad` marks the program's own inputs, and a metadata
    /// view (`reshape` / `permute` / `expand` / `stride`) over one is
    /// still a window onto the same bytes, so the walk follows `source`
    /// to the root exactly as `compute_owner_map` does. Termination is
    /// guaranteed because a view's source is always an earlier node.
    ///
    /// Read this before choosing any tensor as an in-place destination.
    /// Reading a borrowed buffer is fine; writing to one hands the
    /// caller back a mutated argument. Ownership is not visible in the
    /// runtime `chelis_tensor` at all: `chelis_alloc_view` sets
    /// `owns_data = 0` for every intermediate view as well, so the
    /// runtime flag cannot distinguish the two and this plan-level fact
    /// is the only place the distinction exists.
    pub fn borrows_caller_storage(&self, id: NodeId) -> bool {
        let mut cursor = id;
        loop {
            match &self.node_kinds[cursor.0] {
                NodeMemoryKind::BorrowedLoad => return true,
                NodeMemoryKind::MetadataView { source }
                | NodeMemoryKind::StandaloneStore { source }
                | NodeMemoryKind::TerminalDrop { source } => cursor = *source,
                NodeMemoryKind::SlotBacked { .. } | NodeMemoryKind::Skipped => return false,
            }
        }
    }

    pub fn slot(&self, id: usize) -> &SlotPlan {
        &self.slots[id]
    }

    pub fn slots(&self) -> &[SlotPlan] {
        &self.slots
    }

    pub fn emit_cleanup(&self, output_ids: &[NodeId]) -> Vec<String> {
        let mut lines = Vec::new();
        for (idx, kind) in self.node_kinds.iter().enumerate() {
            let id = NodeId(idx);
            if output_ids.contains(&id)
                || matches!(
                    kind,
                    NodeMemoryKind::BorrowedLoad
                        | NodeMemoryKind::TerminalDrop { .. }
                        | NodeMemoryKind::Skipped
                )
            {
                continue;
            }
            lines.push(format!("    chelis_free(t{idx});"));
        }
        for slot in &self.slots {
            lines.push(format!("    chelis_free(chelis_slot{});", slot.id));
        }
        lines
    }
}

fn classify_nodes(dag: &Dag, skipped: &HashSet<NodeId>) -> Vec<NodeMemoryKind> {
    let mut kinds = Vec::with_capacity(dag.len());
    for node in dag.nodes() {
        let kind = if skipped.contains(&node.id) {
            NodeMemoryKind::Skipped
        } else {
            match &node.op {
                RiscOp::Load { .. } => NodeMemoryKind::BorrowedLoad,
                RiscOp::Reshape { .. }
                | RiscOp::Permute { .. }
                | RiscOp::Expand { .. }
                | RiscOp::Stride { .. } => NodeMemoryKind::MetadataView {
                    source: node.inputs[0],
                },
                RiscOp::Drop => NodeMemoryKind::TerminalDrop {
                    source: node.inputs[0],
                },
                RiscOp::Store { .. } => NodeMemoryKind::StandaloneStore {
                    source: node.inputs[0],
                },
                RiscOp::Const { .. }
                | RiscOp::ConstTensor { .. }
                // `Shape` reads only its input's `->shape[axis]` metadata
                // and materializes a fresh rank-0 scalar slot; it does not
                // alias the input buffer (chelis#513/#558).
                | RiscOp::Shape { .. }
                | RiscOp::Add
                | RiscOp::Sub
                | RiscOp::Mul
                | RiscOp::Div
                | RiscOp::FloorDiv
                | RiscOp::TruncDiv
                | RiscOp::CmpLt
                | RiscOp::MaxElem
                | RiscOp::MinElem
                | RiscOp::ExtremaAdjoint { .. }
                | RiscOp::Neg
                | RiscOp::Recip
                | RiscOp::Exp
                | RiscOp::Log
                | RiscOp::Sin
                | RiscOp::Sqrt
                | RiscOp::Cos
                | RiscOp::Tan
                | RiscOp::Atan
                | RiscOp::Abs
                | RiscOp::Floor
                | RiscOp::Ceil
                | RiscOp::Round
                | RiscOp::UniformLike { .. }
                | RiscOp::Dropout { .. }
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
                | RiscOp::Realize
                | RiscOp::Cast { .. }
                | RiscOp::CastTrunc { .. }
                | RiscOp::FusedElem { .. }
                | RiscOp::OneHot { .. }
                | RiscOp::Pad { .. }
                | RiscOp::Shrink { .. }
                | RiscOp::BlasMatmul { .. }
                | RiscOp::Gather { .. }
                | RiscOp::ScatterAdd { .. }
                | RiscOp::Scatter { .. }
                | RiscOp::ScatterElements { .. } => NodeMemoryKind::SlotBacked { slot: usize::MAX },
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
            NodeMemoryKind::SlotBacked { .. } => Some(node.id),
            NodeMemoryKind::BorrowedLoad => None,
            NodeMemoryKind::MetadataView { source }
            | NodeMemoryKind::StandaloneStore { source } => owners[source.0],
            NodeMemoryKind::TerminalDrop { .. } => None,
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
        if matches!(node_kinds[node.id.0], NodeMemoryKind::SlotBacked { .. }) {
            let capacity_elems = logical_elements(&node.output_type);
            by_owner.insert(
                node.id,
                OwnerRequirement {
                    owner: node.id,
                    birth_index: node.id.0,
                    last_use_index: node.id.0,
                    capacity_concrete: capacity_elems.as_concrete(),
                    capacity_key: capacity_elems.normalized_key(),
                    capacity_elems,
                    dtype: node.output_type.precision,
                },
            );
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
            RiscOp::Drop => {
                if let NodeMemoryKind::TerminalDrop { source } = node_kinds[node.id.0] {
                    vec![source]
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
        if matches!(output_node.op, RiscOp::Load { .. } | RiscOp::Store { .. }) {
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
    let mut availability = Vec::<usize>::new();
    let mut slot_capacity_concretes = Vec::<Option<usize>>::new();
    let mut slot_capacity_keys = Vec::<DimExprKey>::new();
    let mut owner_to_slot = HashMap::<NodeId, usize>::new();

    for req in requirements {
        let reused = slots.iter().enumerate().find_map(|(slot_id, slot)| {
            let reusable = slot.dtype == req.dtype
                && availability[slot_id] < req.birth_index
                && capacity_fits(
                    slot_capacity_concretes[slot_id],
                    &slot_capacity_keys[slot_id],
                    req.capacity_concrete,
                    &req.capacity_key,
                );
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
            slot_capacity_concretes.push(req.capacity_concrete);
            slot_capacity_keys.push(req.capacity_key.clone());
            slot_id
        });

        availability[slot_id] = req.last_use_index;
        slots[slot_id].last_use_index = slots[slot_id].last_use_index.max(req.last_use_index);
        owner_to_slot.insert(req.owner, slot_id);
    }

    for (idx, kind) in node_kinds.iter_mut().enumerate() {
        if let NodeMemoryKind::SlotBacked { slot } = kind {
            *slot = owner_to_slot[&NodeId(idx)];
        }
    }

    slots
}

fn capacity_fits(
    slot_concrete: Option<usize>,
    slot_key: &DimExprKey,
    req_concrete: Option<usize>,
    req_key: &DimExprKey,
) -> bool {
    match (slot_concrete, req_concrete) {
        (Some(slot_elems), Some(req_elems)) => slot_elems >= req_elems,
        _ => slot_key == req_key,
    }
}

fn logical_elements(ty: &TensorType) -> DimExpr {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_f32(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: Prim::F32,
        }
    }

    fn sym_f32(name: &str) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Named(name.to_string(), None)],
            precision: Prim::F32,
        }
    }

    fn tensor_f32(dims: Vec<DimInfo>) -> TensorType {
        TensorType {
            dims,
            precision: Prim::F32,
        }
    }

    fn build_plan(dag: &Dag, output_ids: &[NodeId]) -> MemoryPlan {
        MemoryPlan::build(dag, output_ids, &HashSet::new())
    }

    #[test]
    fn planner_reuses_non_overlapping_slots() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let b = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
        let c = dag.add_node(RiscOp::Neg, vec![b], vec_f32(4), None);
        dag.add_root(c);

        let plan = build_plan(&dag, &[c]);
        assert_eq!(plan.slots().len(), 2);
        assert_eq!(plan.node_kind(a), &NodeMemoryKind::SlotBacked { slot: 0 });
        assert_eq!(plan.node_kind(c), &NodeMemoryKind::SlotBacked { slot: 0 });
        assert_eq!(plan.node_kind(b), &NodeMemoryKind::SlotBacked { slot: 1 });
    }

    #[test]
    fn planner_keeps_overlapping_values_separate() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        let c = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4), None);
        dag.add_root(c);

        let plan = build_plan(&dag, &[c]);
        assert_eq!(plan.slots().len(), 3);
    }

    #[test]
    fn planner_keeps_loads_borrowed() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let y = dag.add_node(RiscOp::Neg, vec![x], vec_f32(4), None);
        dag.add_root(y);

        let plan = build_plan(&dag, &[y]);
        assert_eq!(plan.node_kind(x), &NodeMemoryKind::BorrowedLoad);
        assert_eq!(plan.slots().len(), 1);
        assert!(
            !plan
                .emit_cleanup(&[y])
                .iter()
                .any(|line| line == "    chelis_free(t0);")
        );
    }

    // chelis#933: `borrows_caller_storage` is what stops an in-place
    // optimization from writing into a buffer the caller owns, so it
    // needs both directions pinned — a false positive silently deletes
    // the optimization, a false negative silently corrupts an argument.
    #[test]
    fn borrows_caller_storage_reports_program_inputs() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let y = dag.add_node(RiscOp::Neg, vec![x], vec_f32(4), None);
        dag.add_root(y);

        let plan = build_plan(&dag, &[y]);
        assert!(
            plan.borrows_caller_storage(x),
            "a program input's bytes belong to the caller"
        );
        assert!(
            !plan.borrows_caller_storage(y),
            "an operator result lives in a slot this program allocated"
        );
    }

    #[test]
    fn borrows_caller_storage_sees_through_a_view_of_an_input() {
        // A `reshape`/`expand`/`permute` of an input is a metadata view:
        // different shape, same bytes. Checking only the node itself
        // would miss it.
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
        let v = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::Concrete(4),
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(4), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let y = dag.add_node(RiscOp::Neg, vec![v], vec_f32(4), None);
        dag.add_root(y);

        let plan = build_plan(&dag, &[y]);
        assert_eq!(
            plan.node_kind(v),
            &NodeMemoryKind::MetadataView { source: x }
        );
        assert!(
            plan.borrows_caller_storage(v),
            "a view of a program input is still a window onto the caller's bytes"
        );
    }

    #[test]
    fn borrows_caller_storage_clears_a_view_of_an_owned_intermediate() {
        // Negative parity for the case above: the same view shape over a
        // program-owned node must stay reusable, or the fix would delete
        // in-place fusion everywhere rather than only where it was wrong.
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let v = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::Concrete(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(4), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let y = dag.add_node(RiscOp::Neg, vec![v], vec_f32(4), None);
        dag.add_root(y);

        let plan = build_plan(&dag, &[y]);
        assert!(
            !plan.borrows_caller_storage(v),
            "a view of an owned intermediate is program-owned storage"
        );
    }

    #[test]
    fn metadata_view_extends_source_lifetime() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let v = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::Concrete(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(4), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        let c = dag.add_node(RiscOp::Neg, vec![v], vec_f32(4), None);
        dag.add_root(c);

        let plan = build_plan(&dag, &[c]);
        assert_eq!(
            plan.node_kind(v),
            &NodeMemoryKind::MetadataView { source: a }
        );
        assert_ne!(plan.node_kind(a), plan.node_kind(b));
    }

    #[test]
    fn store_outputs_are_standalone_not_slots() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let s = dag.add_node(
            RiscOp::Store { name: "out".into() },
            vec![a],
            vec_f32(4),
            None,
        );
        dag.add_root(s);

        let plan = build_plan(&dag, &[s]);
        assert_eq!(
            plan.node_kind(s),
            &NodeMemoryKind::StandaloneStore { source: a }
        );
        assert_eq!(plan.slots().len(), 1);
    }

    #[test]
    fn terminal_drop_closes_slot_lifetime_for_reuse() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let drop = dag.add_node(RiscOp::Drop, vec![a], vec_f32(4), None);
        let b = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 2.0),
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_root(b);

        let plan = build_plan(&dag, &[b]);
        assert_eq!(
            plan.node_kind(drop),
            &NodeMemoryKind::TerminalDrop { source: a }
        );
        assert_eq!(
            plan.node_kind(a),
            plan.node_kind(b),
            "Drop must make a's slot available before b is born"
        );
        assert!(
            !plan
                .emit_cleanup(&[b])
                .iter()
                .any(|line| line.contains(&format!("t{}", drop.0))),
            "Drop is terminal metadata and must not allocate a wrapper"
        );
    }

    #[test]
    fn symbolic_exact_match_reuses_but_non_match_falls_back() {
        let mut exact = Dag::new();
        let a = exact.add_node(
            RiscOp::synth_const(sym_f32("n").precision, 1.0),
            vec![],
            sym_f32("n"),
            None,
        );
        let b = exact.add_node(RiscOp::Neg, vec![a], sym_f32("n"), None);
        let c = exact.add_node(RiscOp::Neg, vec![b], sym_f32("n"), None);
        exact.add_root(c);
        assert_eq!(build_plan(&exact, &[c]).slots().len(), 2);

        let mut mismatch = Dag::new();
        let a = mismatch.add_node(
            RiscOp::synth_const(sym_f32("m").precision, 1.0),
            vec![],
            sym_f32("m"),
            None,
        );
        let _b = mismatch.add_node(RiscOp::Neg, vec![a], sym_f32("m"), None);
        let c = mismatch.add_node(
            RiscOp::synth_const(sym_f32("n").precision, 2.0),
            vec![],
            sym_f32("n"),
            None,
        );
        mismatch.add_root(c);
        assert_eq!(build_plan(&mismatch, &[c]).slots().len(), 3);
    }

    #[test]
    fn planner_reuses_commuted_symbolic_product_capacity() {
        let mn = tensor_f32(vec![
            DimInfo::Named("m".into(), None),
            DimInfo::Named("n".into(), None),
        ]);
        let nm = tensor_f32(vec![
            DimInfo::Named("n".into(), None),
            DimInfo::Named("m".into(), None),
        ]);
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(mn.precision, 1.0),
            vec![],
            mn.clone(),
            None,
        );
        let b = dag.add_node(RiscOp::Neg, vec![a], mn, None);
        let c = dag.add_node(RiscOp::Neg, vec![b], nm, None);
        dag.add_root(c);

        let plan = build_plan(&dag, &[c]);
        assert_eq!(
            plan.slots().len(),
            2,
            "dead m*n capacity should fit later n*m requirement"
        );
        assert_eq!(plan.node_kind(a), plan.node_kind(c));
    }

    #[test]
    fn planner_reuses_identity_simplified_symbolic_capacity() {
        let n = sym_f32("n");
        let n_by_one = tensor_f32(vec![DimInfo::Named("n".into(), None), DimInfo::Lit(1)]);
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(n_by_one.precision, 1.0),
            vec![],
            n_by_one,
            None,
        );
        let b = dag.add_node(RiscOp::Neg, vec![a], n.clone(), None);
        let c = dag.add_node(RiscOp::Neg, vec![b], n, None);
        dag.add_root(c);

        let plan = build_plan(&dag, &[c]);
        assert_eq!(
            plan.slots().len(),
            2,
            "n*1 capacity should fit later n requirement"
        );
        assert_eq!(plan.node_kind(a), plan.node_kind(c));
    }

    #[test]
    fn planner_does_not_alpha_rename_unrelated_symbolic_capacity() {
        let mn = tensor_f32(vec![
            DimInfo::Named("m".into(), None),
            DimInfo::Named("n".into(), None),
        ]);
        let xy = tensor_f32(vec![
            DimInfo::Named("x".into(), None),
            DimInfo::Named("y".into(), None),
        ]);
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(mn.precision, 1.0),
            vec![],
            mn.clone(),
            None,
        );
        let b = dag.add_node(RiscOp::Neg, vec![a], mn, None);
        let c = dag.add_node(RiscOp::Neg, vec![b], xy, None);
        dag.add_root(c);

        let plan = build_plan(&dag, &[c]);
        assert_eq!(
            plan.slots().len(),
            3,
            "same-shaped symbolic products are not equivalent without \
             explicit binder identity"
        );
        assert_ne!(plan.node_kind(a), plan.node_kind(c));
    }

    #[test]
    fn capacity_fits_concrete_larger_and_rejects_distinct_symbolic_keys() {
        let slot = DimExpr::Concrete(8);
        let req = DimExpr::Concrete(4);
        assert!(capacity_fits(
            slot.as_concrete(),
            &slot.normalized_key(),
            req.as_concrete(),
            &req.normalized_key()
        ));

        let slot = DimExpr::Mul(
            Box::new(DimExpr::Sym("m".into())),
            Box::new(DimExpr::Sym("n".into())),
        );
        let req = DimExpr::Mul(
            Box::new(DimExpr::Sym("m".into())),
            Box::new(DimExpr::Sym("k".into())),
        );
        assert!(!capacity_fits(
            slot.as_concrete(),
            &slot.normalized_key(),
            req.as_concrete(),
            &req.normalized_key()
        ));
    }

    #[test]
    fn cleanup_frees_wrappers_at_epilogue_then_slots() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(vec_f32(4).precision, 1.0),
            vec![],
            vec_f32(4),
            None,
        );
        let b = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
        dag.add_root(b);

        let plan = build_plan(&dag, &[b]);
        let lines = plan.emit_cleanup(&[b]);
        assert_eq!(lines[0], "    chelis_free(t0);");
        assert!(lines[1].starts_with("    chelis_free(chelis_slot"));
        assert!(lines[2].starts_with("    chelis_free(chelis_slot"));
    }
}
