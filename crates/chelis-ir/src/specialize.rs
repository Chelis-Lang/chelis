//! IR-level backend specialization.
//!
//! This pass runs after AD and before DCE/fusion/codegen. It is deliberately
//! conservative: the no-op cleanup has a closed list, and specialization
//! replaces recognized subgraphs with explicit backend-specialized IR nodes.

use std::collections::HashMap;

use crate::dag::{Dag, DagNode, DimExpr, NodeId, RiscOp, TensorType};

/// Run the closed-list no-op cleanup plus backend specialization, then DCE.
pub fn specialize_for_blas(dag: &Dag) -> Dag {
    let cleaned = eliminate_closed_list_noops(dag);
    let specialized = replace_matmul_patterns(&cleaned);
    crate::optimize::dead_code_eliminate(&specialized)
}

/// Eliminate only the M1 closed-list no-ops:
/// identity Cast, identity Reshape, and identity Permute.
pub fn eliminate_closed_list_noops(dag: &Dag) -> Dag {
    let mut out = Dag::new();
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in dag.nodes() {
        let remapped_inputs: Vec<NodeId> = node.inputs.iter().map(|id| id_map[id]).collect();
        if let Some(source) = identity_source(node, dag) {
            let mapped = id_map[&source];
            id_map.insert(node.id, mapped);
            append_node_provenance(&mut out, mapped, node);
            continue;
        }

        let new_id = out.add_node(
            node.op.clone(),
            remapped_inputs,
            node.output_type.clone(),
            node.span_id.clone(),
        );
        if let Some(reusable) = node.reusable_input
            && let Some(mapped) = id_map.get(&reusable).copied()
        {
            out.set_reusable_input(new_id, mapped);
        }
        if !node.merged_spans.is_empty()
            && let Some(new_node) = out.node_mut(new_id)
        {
            new_node.merged_spans = node.merged_spans.clone();
        }
        id_map.insert(node.id, new_id);
    }

    for &root in dag.roots() {
        if let Some(&mapped) = id_map.get(&root) {
            out.add_root(mapped);
        }
    }
    out
}

fn identity_source(node: &DagNode, dag: &Dag) -> Option<NodeId> {
    if node.inputs.len() != 1 {
        return None;
    }
    let input_id = node.inputs[0];
    let input = dag.get(input_id)?;
    match &node.op {
        RiscOp::Cast { new_precision }
            if input.output_type.precision == *new_precision
                && input.output_type.dims == node.output_type.dims =>
        {
            Some(input_id)
        }
        RiscOp::Reshape { new_shape }
            if input.output_type.dims == *new_shape
                && input.output_type.dims == node.output_type.dims =>
        {
            Some(input_id)
        }
        RiscOp::Permute { axes }
            if axes.iter().copied().eq(0..axes.len())
                && input.output_type.dims == node.output_type.dims =>
        {
            Some(input_id)
        }
        _ => None,
    }
}

fn replace_matmul_patterns(dag: &Dag) -> Dag {
    let mut out = Dag::new();
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in dag.nodes() {
        if let Some(info) = detect_matmul_pattern(dag, node.id) {
            let a = id_map[&info.a];
            let b = id_map[&info.b];
            let new_id = out.add_node(
                RiscOp::BlasMatmul {
                    batch_dims: info.batch_dims.clone(),
                    m: info.m.clone(),
                    n: info.n.clone(),
                    k: info.k.clone(),
                },
                vec![a, b],
                node.output_type.clone(),
                node.span_id.clone(),
            );
            append_consumed_provenance(&mut out, new_id, dag, node, &info);
            id_map.insert(node.id, new_id);
            continue;
        }

        let remapped_inputs: Vec<NodeId> = node.inputs.iter().map(|id| id_map[id]).collect();
        let new_id = out.add_node(
            node.op.clone(),
            remapped_inputs,
            node.output_type.clone(),
            node.span_id.clone(),
        );
        if let Some(reusable) = node.reusable_input
            && let Some(mapped) = id_map.get(&reusable).copied()
        {
            out.set_reusable_input(new_id, mapped);
        }
        if !node.merged_spans.is_empty()
            && let Some(new_node) = out.node_mut(new_id)
        {
            new_node.merged_spans = node.merged_spans.clone();
        }
        id_map.insert(node.id, new_id);
    }

    for &root in dag.roots() {
        if let Some(&mapped) = id_map.get(&root) {
            out.add_root(mapped);
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MatmulInfo {
    a: NodeId,
    b: NodeId,
    batch_dims: Vec<DimExpr>,
    m: DimExpr,
    n: DimExpr,
    k: DimExpr,
    mul: NodeId,
    expand_a: NodeId,
    expand_b: NodeId,
}

fn detect_matmul_pattern(dag: &Dag, sum_id: NodeId) -> Option<MatmulInfo> {
    let sum_node = dag.get(sum_id)?;
    let sum_axis = match &sum_node.op {
        RiscOp::Sum { axis } => *axis,
        _ => return None,
    };
    if sum_node.inputs.len() != 1 || sum_node.output_type.dims.len() < 2 {
        return None;
    }
    let lead_len = sum_node.output_type.dims.len() - 2;
    if sum_axis != lead_len + 1 {
        return None;
    }
    let mul_id = sum_node.inputs[0];
    let mul_node = dag.get(mul_id)?;
    if !matches!(mul_node.op, RiscOp::Mul) || mul_node.inputs.len() != 2 {
        return None;
    }
    let expand_a_id = mul_node.inputs[0];
    let expand_b_id = mul_node.inputs[1];
    let expand_a = dag.get(expand_a_id)?;
    let expand_b = dag.get(expand_b_id)?;
    let RiscOp::Expand { axis: axis_a, .. } = expand_a.op else {
        return None;
    };
    let RiscOp::Expand { axis: axis_b, .. } = expand_b.op else {
        return None;
    };
    if axis_a != lead_len + 2 || axis_b != lead_len {
        return None;
    }
    if expand_a.inputs.len() != 1 || expand_b.inputs.len() != 1 {
        return None;
    }

    let a = expand_a.inputs[0];
    let b = expand_b.inputs[0];
    if !node_has_contiguous_matrix_slices(dag, a, 2)
        || !node_has_contiguous_matrix_slices(dag, b, 2)
    {
        return None;
    }
    let a_ty = &dag.get(a)?.output_type;
    let b_ty = &dag.get(b)?.output_type;
    let batch_dims = sum_node.output_type.dims[..lead_len]
        .iter()
        .map(DimExpr::from)
        .collect::<Vec<_>>();
    let (m, n, k) = matmul_dims(a_ty, b_ty, &batch_dims)?;
    let out_dims = sum_node
        .output_type
        .dims
        .iter()
        .map(DimExpr::from)
        .collect::<Vec<_>>();
    let expected_out = batch_dims
        .iter()
        .cloned()
        .chain([m.clone(), n.clone()])
        .collect::<Vec<_>>();
    if out_dims != expected_out {
        return None;
    }
    Some(MatmulInfo {
        a,
        b,
        batch_dims,
        m,
        n,
        k,
        mul: mul_id,
        expand_a: expand_a_id,
        expand_b: expand_b_id,
    })
}

fn matmul_dims(
    a_ty: &TensorType,
    b_ty: &TensorType,
    batch_dims: &[DimExpr],
) -> Option<(DimExpr, DimExpr, DimExpr)> {
    if a_ty.dims.len() != batch_dims.len() + 2 || b_ty.dims.len() != batch_dims.len() + 2 {
        return None;
    }
    let a_dims = a_ty.dims.iter().map(DimExpr::from).collect::<Vec<_>>();
    let b_dims = b_ty.dims.iter().map(DimExpr::from).collect::<Vec<_>>();
    if &a_dims[..batch_dims.len()] != batch_dims || &b_dims[..batch_dims.len()] != batch_dims {
        return None;
    }
    let m = a_dims[a_dims.len() - 2].clone();
    let k_a = a_dims[a_dims.len() - 1].clone();
    let k_b = b_dims[b_dims.len() - 2].clone();
    let n = b_dims[b_dims.len() - 1].clone();
    if k_a == k_b { Some((m, n, k_a)) } else { None }
}

fn node_has_contiguous_matrix_slices(dag: &Dag, id: NodeId, matrix_rank: usize) -> bool {
    let Some(node) = dag.get(id) else {
        return false;
    };
    match &node.op {
        RiscOp::Load { .. }
        | RiscOp::Const { .. }
        | RiscOp::Add
        | RiscOp::Mul
        | RiscOp::MaxElem
        | RiscOp::CmpLt
        | RiscOp::Neg
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
        | RiscOp::UniformLike { .. }
        | RiscOp::Dropout { .. }
        | RiscOp::Copy
        | RiscOp::Drop
        | RiscOp::Sum { .. }
        | RiscOp::MaxReduce { .. }
        | RiscOp::MinReduce { .. }
        | RiscOp::ProdReduce { .. }
        | RiscOp::Argmax { .. }
        | RiscOp::Argmin { .. }
        | RiscOp::Realize
        | RiscOp::Cast { .. }
        | RiscOp::FusedElem { .. }
        | RiscOp::BlasMatmul { .. }
        | RiscOp::Gather { .. }
        | RiscOp::ScatterAdd { .. } => true,
        RiscOp::Reshape { .. } | RiscOp::Store { .. } => node
            .inputs
            .first()
            .copied()
            .is_some_and(|input| node_has_contiguous_matrix_slices(dag, input, matrix_rank)),
        RiscOp::Expand { axis, .. } => {
            let rank = node.output_type.dims.len();
            *axis < rank.saturating_sub(matrix_rank)
                && node
                    .inputs
                    .first()
                    .copied()
                    .is_some_and(|input| node_has_contiguous_matrix_slices(dag, input, matrix_rank))
        }
        RiscOp::Permute { .. }
        | RiscOp::Stride { .. }
        | RiscOp::Pad { .. }
        | RiscOp::Shrink { .. } => false,
    }
}

fn append_consumed_provenance(
    out: &mut Dag,
    target: NodeId,
    dag: &Dag,
    sum_node: &DagNode,
    info: &MatmulInfo,
) {
    append_node_provenance(out, target, sum_node);
    for id in [info.mul, info.expand_a, info.expand_b] {
        if let Some(node) = dag.get(id) {
            append_node_provenance(out, target, node);
        }
    }
}

fn append_node_provenance(out: &mut Dag, target: NodeId, source: &DagNode) {
    crate::span_merge::append_span_to_node(out, target, source.span_id.as_deref());
    crate::span_merge::append_spans_to_node(out, target, &source.merged_spans);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{Dag, DimExpr, DimInfo, TensorType};
    use chelis_types::types::Prim;

    fn mat(r: usize, c: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
            precision: Prim::F32,
        }
    }

    fn t3(a: usize, b: usize, c: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
            precision: Prim::F32,
        }
    }

    fn symbolic_mat(r: &str, c: &str) -> TensorType {
        TensorType {
            dims: vec![
                DimInfo::Named(r.to_string(), None),
                DimInfo::Named(c.to_string(), None),
            ],
            precision: Prim::F32,
        }
    }

    fn symbolic_t3(a: &str, b: &str, c: &str) -> TensorType {
        TensorType {
            dims: vec![
                DimInfo::Named(a.to_string(), None),
                DimInfo::Named(b.to_string(), None),
                DimInfo::Named(c.to_string(), None),
            ],
            precision: Prim::F32,
        }
    }

    fn symbolic_t4(a: &str, b: &str, c: &str, d: &str) -> TensorType {
        TensorType {
            dims: vec![
                DimInfo::Named(a.to_string(), None),
                DimInfo::Named(b.to_string(), None),
                DimInfo::Named(c.to_string(), None),
                DimInfo::Named(d.to_string(), None),
            ],
            precision: Prim::F32,
        }
    }

    #[test]
    fn identity_cast_does_not_hide_matmul() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat(2, 3), None);
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat(3, 4), None);
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: DimExpr::Concrete(4),
            },
            vec![a],
            t3(2, 3, 4),
            None,
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::Concrete(2),
            },
            vec![b],
            t3(2, 3, 4),
            None,
        );
        let ca = dag.add_node(
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            vec![ea],
            t3(2, 3, 4),
            None,
        );
        let cb = dag.add_node(
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            vec![eb],
            t3(2, 3, 4),
            None,
        );
        let mul = dag.add_node(RiscOp::Mul, vec![ca, cb], t3(2, 3, 4), None);
        let sum = dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat(2, 4), None);
        dag.add_root(sum);

        let out = specialize_for_blas(&dag);
        assert!(out.nodes().iter().any(|node| {
            matches!(
                &node.op,
                RiscOp::BlasMatmul {
                    batch_dims,
                    m,
                    n,
                    k,
                } if batch_dims.is_empty()
                    && *m == DimExpr::Concrete(2)
                    && *n == DimExpr::Concrete(4)
                    && *k == DimExpr::Concrete(3)
            )
        }));
        assert!(
            !out.nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Mul | RiscOp::Expand { .. }))
        );
    }

    #[test]
    fn symbolic_matmul_specializes_to_runtime_dim_blas() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            symbolic_mat("m", "k"),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            symbolic_mat("k", "n"),
            None,
        );
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: DimExpr::Sym("n".into()),
            },
            vec![a],
            symbolic_t3("m", "k", "n"),
            None,
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::Sym("m".into()),
            },
            vec![b],
            symbolic_t3("m", "k", "n"),
            None,
        );
        let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], symbolic_t3("m", "k", "n"), None);
        let sum = dag.add_node(
            RiscOp::Sum { axis: 1 },
            vec![mul],
            symbolic_mat("m", "n"),
            None,
        );
        dag.add_root(sum);

        let out = specialize_for_blas(&dag);
        assert!(out.nodes().iter().any(|node| {
            matches!(
                &node.op,
                RiscOp::BlasMatmul {
                    batch_dims,
                    m,
                    n,
                    k,
                } if batch_dims.is_empty()
                    && *m == DimExpr::Sym("m".into())
                    && *n == DimExpr::Sym("n".into())
                    && *k == DimExpr::Sym("k".into())
            )
        }));
        assert!(
            !out.nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Mul | RiscOp::Expand { .. })),
            "symbolic BLAS specialization should DCE the dense product path"
        );
    }

    #[test]
    fn noncontiguous_operand_stays_on_generic_path() {
        let mut dag = Dag::new();
        let base_a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat(3, 2), None);
        let a = dag.add_node(
            RiscOp::Permute { axes: vec![1, 0] },
            vec![base_a],
            mat(2, 3),
            None,
        );
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat(3, 4), None);
        let ea = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: DimExpr::Concrete(4),
            },
            vec![a],
            t3(2, 3, 4),
            None,
        );
        let eb = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::Concrete(2),
            },
            vec![b],
            t3(2, 3, 4),
            None,
        );
        let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], t3(2, 3, 4), None);
        let sum = dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat(2, 4), None);
        dag.add_root(sum);

        let out = specialize_for_blas(&dag);
        assert!(
            !out.nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::BlasMatmul { .. })),
            "non-contiguous matmul operands must stay on the generic path"
        );
    }

    #[test]
    fn rank4_symbolic_batched_matmul_specializes() {
        let mut dag = Dag::new();
        let a_ty = symbolic_t4("batch", "heads", "seq", "dim");
        let b_ty = symbolic_t4("batch", "heads", "dim", "seq");
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            a_ty.clone(),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            b_ty.clone(),
            None,
        );
        let out = crate::tier2::lower_matmul(&mut dag, a, b, &a_ty, &b_ty, None);
        dag.add_root(out);

        let specialized = specialize_for_blas(&dag);
        assert!(specialized.nodes().iter().any(|node| {
            matches!(
                &node.op,
                RiscOp::BlasMatmul {
                    batch_dims,
                    m,
                    n,
                    k,
                } if batch_dims == &vec![DimExpr::Sym("batch".into()), DimExpr::Sym("heads".into())]
                    && *m == DimExpr::Sym("seq".into())
                    && *n == DimExpr::Sym("seq".into())
                    && *k == DimExpr::Sym("dim".into())
            )
        }));
        assert!(
            !specialized
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Mul | RiscOp::Expand { .. })),
            "batched BLAS specialization must remove the dense [..., m, k, n] product"
        );
    }

    #[test]
    fn terminal_drop_markers_do_not_block_matmul_specialization() {
        let mut dag = Dag::new();
        let a_ty = mat(8, 16);
        let b_ty = mat(16, 4);
        let a = dag.add_node(
            RiscOp::Load { name: "a".into() },
            vec![],
            a_ty.clone(),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "b".into() },
            vec![],
            b_ty.clone(),
            None,
        );
        let out = crate::tier2::lower_matmul(&mut dag, a, b, &a_ty, &b_ty, None);
        let mul = dag
            .nodes()
            .iter()
            .find(|node| matches!(node.op, RiscOp::Mul))
            .expect("tier2 matmul contains mul")
            .id;
        dag.add_node(RiscOp::Drop, vec![mul], t3(8, 16, 4), None);
        dag.add_root(out);

        let specialized = specialize_for_blas(&dag);
        let fused = crate::fuse::fuse(&specialized);
        assert!(
            fused
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::BlasMatmul { .. })),
            "{:?}",
            fused.nodes()
        );
    }
}
