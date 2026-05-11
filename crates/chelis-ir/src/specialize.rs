//! IR-level backend specialization.
//!
//! This pass runs after AD and before DCE/fusion/codegen. It is deliberately
//! conservative: the no-op cleanup has a closed list, and specialization
//! replaces recognized subgraphs with explicit backend-specialized IR nodes.

use std::collections::HashMap;

use crate::dag::{Dag, DagNode, DimExpr, DimInfo, NodeId, RiscOp, TensorType};
use chelis_types::types::Prim;

/// Compiler pipeline ordering around backend specialization.
///
/// This is intentionally represented as code, not only prose, because gather /
/// scatter recognizers, cross-function specialization, and future in-place
/// rewrites all depend on the same ordering contract.
pub const SPECIALIZATION_PIPELINE_ORDER: &[&str] = &[
    "ad",
    "no_op_cleanup",
    "dense_gather_recognizer",
    "one_hot_fallback_lowering",
    "blas_matmul_recognizer",
    "cross_function_specialization",
    "dce",
    "in_place_fusion",
    "codegen",
];

/// Run the closed-list no-op cleanup plus backend specialization, then DCE.
pub fn specialize_for_blas(dag: &Dag) -> Dag {
    let cleaned = eliminate_closed_list_noops(dag);
    let gathered = replace_dense_gather_patterns(&cleaned);
    let lowered = lower_unmatched_one_hot(&gathered);
    let specialized = replace_matmul_patterns(&lowered);
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
            // The matmul-pattern detector recognizes the
            // `Sum(Mul(Expand(A), Expand(B)))` shape from tier2 lowering.
            // The accumulator on the synthesized BlasMatmul follows the
            // accumulator pinned on the source `Sum` node so the WS-A0
            // §5.7.1 default propagates through specialization.
            let accumulator = match &dag.get(node.id).map(|n| &n.op) {
                Some(RiscOp::Sum { accumulator, .. }) => *accumulator,
                _ => node.output_type.precision,
            };
            let new_id = out.add_node(
                RiscOp::BlasMatmul {
                    batch_dims: info.batch_dims.clone(),
                    m: info.m.clone(),
                    n: info.n.clone(),
                    k: info.k.clone(),
                    accumulator,
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

fn replace_dense_gather_patterns(dag: &Dag) -> Dag {
    let mut out = Dag::new();
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in dag.nodes() {
        if let Some(info) = detect_dense_gather_pattern(dag, node.id) {
            let values = id_map[&info.values];
            let indices = id_map[&info.indices];
            let new_id = out.add_node(
                RiscOp::Gather { axis: 0 },
                vec![values, indices],
                node.output_type.clone(),
                node.span_id.clone(),
            );
            append_dense_gather_provenance(&mut out, new_id, dag, node, &info);
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

fn lower_unmatched_one_hot(dag: &Dag) -> Dag {
    let mut out = Dag::new();
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in dag.nodes() {
        if let RiscOp::OneHot { vocab } = node.op {
            let indices = id_map[&node.inputs[0]];
            let new_id = lower_one_hot_node(&mut out, indices, node, vocab);
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

fn lower_one_hot_node(out: &mut Dag, indices: NodeId, source: &DagNode, vocab: usize) -> NodeId {
    assert!(vocab > 0, "one_hot vocab must be nonzero");
    let indices_ty = out
        .get(indices)
        .expect("one_hot input must have been remapped")
        .output_type
        .clone();
    let bool_ty = TensorType {
        dims: indices_ty.dims.clone(),
        precision: Prim::Bool,
    };
    let col_ty = TensorType {
        dims: indices_ty
            .dims
            .iter()
            .cloned()
            .chain([DimInfo::Lit(1)])
            .collect(),
        precision: Prim::F32,
    };
    let vocab_axis = indices_ty.dims.len();
    let mut accumulated = None;

    for class in 0..vocab {
        let class_id = out.add_node(
            RiscOp::Const {
                value: class as f64,
            },
            vec![],
            indices_ty.clone(),
            source.span_id.clone(),
        );
        let lt_l = out.add_node(
            RiscOp::CmpLt,
            vec![indices, class_id],
            bool_ty.clone(),
            source.span_id.clone(),
        );
        let lt_r = out.add_node(
            RiscOp::CmpLt,
            vec![class_id, indices],
            bool_ty.clone(),
            source.span_id.clone(),
        );
        let neq = out.add_node(
            RiscOp::MaxElem,
            vec![lt_l, lt_r],
            bool_ty.clone(),
            source.span_id.clone(),
        );
        let one = out.add_node(
            RiscOp::Const { value: 1.0 },
            vec![],
            bool_ty.clone(),
            source.span_id.clone(),
        );
        let eq_bool = out.add_node(
            RiscOp::CmpLt,
            vec![neq, one],
            bool_ty.clone(),
            source.span_id.clone(),
        );
        let eq_f32 = out.add_node(
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            vec![eq_bool],
            TensorType {
                dims: indices_ty.dims.clone(),
                precision: Prim::F32,
            },
            source.span_id.clone(),
        );
        let col = out.add_node(
            RiscOp::Expand {
                axis: vocab_axis,
                size: DimExpr::Concrete(1),
            },
            vec![eq_f32],
            col_ty.clone(),
            source.span_id.clone(),
        );
        let padded = out.add_node(
            RiscOp::Pad {
                padding: indices_ty
                    .dims
                    .iter()
                    .map(|_| (0, 0))
                    .chain([(class, vocab - class - 1)])
                    .collect(),
                fill: 0.0,
            },
            vec![col],
            source.output_type.clone(),
            source.span_id.clone(),
        );
        append_node_provenance(out, padded, source);
        accumulated = Some(match accumulated {
            Some(prev) => out.add_node(
                RiscOp::Add,
                vec![prev, padded],
                source.output_type.clone(),
                source.span_id.clone(),
            ),
            None => padded,
        });
    }

    let result = accumulated.expect("nonzero vocab creates at least one column");
    append_node_provenance(out, result, source);
    result
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct DenseGatherInfo {
    values: NodeId,
    indices: NodeId,
    mul: NodeId,
    expand_one_hot: NodeId,
    expand_values: NodeId,
    one_hot: NodeId,
}

fn detect_dense_gather_pattern(dag: &Dag, sum_id: NodeId) -> Option<DenseGatherInfo> {
    let sum_node = dag.get(sum_id)?;
    let sum_axis = match &sum_node.op {
        RiscOp::Sum { axis, .. } => *axis,
        _ => return None,
    };
    if sum_node.inputs.len() != 1 {
        return None;
    }

    let mul_id = sum_node.inputs[0];
    let mul_node = dag.get(mul_id)?;
    if !matches!(mul_node.op, RiscOp::Mul) || mul_node.inputs.len() != 2 {
        return None;
    }

    detect_dense_gather_operands(
        dag,
        sum_node,
        sum_axis,
        mul_id,
        mul_node.inputs[0],
        mul_node.inputs[1],
    )
    .or_else(|| {
        detect_dense_gather_operands(
            dag,
            sum_node,
            sum_axis,
            mul_id,
            mul_node.inputs[1],
            mul_node.inputs[0],
        )
    })
}

fn detect_dense_gather_operands(
    dag: &Dag,
    sum_node: &DagNode,
    sum_axis: usize,
    mul_id: NodeId,
    expand_one_hot_id: NodeId,
    expand_values_id: NodeId,
) -> Option<DenseGatherInfo> {
    let expand_one_hot = dag.get(expand_one_hot_id)?;
    let expand_values = dag.get(expand_values_id)?;
    let RiscOp::Expand {
        axis: one_hot_expand_axis,
        ..
    } = expand_one_hot.op
    else {
        return None;
    };
    let RiscOp::Expand {
        axis: values_expand_axis,
        ..
    } = expand_values.op
    else {
        return None;
    };
    if expand_one_hot.inputs.len() != 1 || expand_values.inputs.len() != 1 {
        return None;
    }

    let one_hot_id = expand_one_hot.inputs[0];
    let one_hot = dag.get(one_hot_id)?;
    let RiscOp::OneHot { vocab } = one_hot.op else {
        return None;
    };
    if one_hot.inputs.len() != 1 {
        return None;
    }
    let indices_id = one_hot.inputs[0];
    let indices_ty = &dag.get(indices_id)?.output_type;
    if indices_ty.dims.len() != 1 {
        return None;
    }
    let values_id = expand_values.inputs[0];
    let values_ty = &dag.get(values_id)?.output_type;
    if values_ty.dims.len() != 2 {
        return None;
    }
    let vocab_axis = indices_ty.dims.len();
    if sum_axis != vocab_axis || one_hot_expand_axis != vocab_axis + 1 || values_expand_axis != 0 {
        return None;
    }
    if one_hot.output_type.dims != vec![indices_ty.dims[0].clone(), DimInfo::Lit(vocab)] {
        return None;
    }
    if !dims_equivalent(&values_ty.dims[0], &DimInfo::Lit(vocab)) {
        return None;
    }
    let expected_expanded = vec![
        indices_ty.dims[0].clone(),
        values_ty.dims[0].clone(),
        values_ty.dims[1].clone(),
    ];
    if expand_one_hot.output_type.dims != expected_expanded
        || expand_values.output_type.dims != expected_expanded
    {
        return None;
    }
    let expected_output = vec![indices_ty.dims[0].clone(), values_ty.dims[1].clone()];
    if sum_node.output_type.dims != expected_output {
        return None;
    }

    Some(DenseGatherInfo {
        values: values_id,
        indices: indices_id,
        mul: mul_id,
        expand_one_hot: expand_one_hot_id,
        expand_values: expand_values_id,
        one_hot: one_hot_id,
    })
}

fn dims_equivalent(lhs: &DimInfo, rhs: &DimInfo) -> bool {
    DimExpr::from(lhs).normalized_key() == DimExpr::from(rhs).normalized_key()
}

fn detect_matmul_pattern(dag: &Dag, sum_id: NodeId) -> Option<MatmulInfo> {
    let sum_node = dag.get(sum_id)?;
    let sum_axis = match &sum_node.op {
        RiscOp::Sum { axis, .. } => *axis,
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
        | RiscOp::OneHot { .. }
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

fn append_dense_gather_provenance(
    out: &mut Dag,
    target: NodeId,
    dag: &Dag,
    sum_node: &DagNode,
    info: &DenseGatherInfo,
) {
    append_node_provenance(out, target, sum_node);
    for id in [
        info.mul,
        info.expand_one_hot,
        info.expand_values,
        info.one_hot,
    ] {
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

    fn vec_i32(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: Prim::Int32,
        }
    }

    #[test]
    fn specialization_pipeline_order_is_pinned() {
        assert_eq!(
            SPECIALIZATION_PIPELINE_ORDER,
            &[
                "ad",
                "no_op_cleanup",
                "dense_gather_recognizer",
                "one_hot_fallback_lowering",
                "blas_matmul_recognizer",
                "cross_function_specialization",
                "dce",
                "in_place_fusion",
                "codegen",
            ]
        );
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
        let sum = dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![mul],
            mat(2, 4),
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
                    ..
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
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
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
                    ..
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
        let sum = dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![mul],
            mat(2, 4),
            None,
        );
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
                    ..
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

    #[test]
    fn one_hot_dense_gather_specializes_before_matmul() {
        let mut dag = Dag::new();
        let values = dag.add_node(
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            mat(2, 3),
            None,
        );
        let indices = dag.add_node(
            RiscOp::Load {
                name: "indices".into(),
            },
            vec![],
            vec_i32(4),
            None,
        );
        let one_hot = dag.add_node(RiscOp::OneHot { vocab: 2 }, vec![indices], mat(4, 2), None);
        let one_hot_exp = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: DimExpr::Concrete(3),
            },
            vec![one_hot],
            t3(4, 2, 3),
            None,
        );
        let values_exp = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::Concrete(4),
            },
            vec![values],
            t3(4, 2, 3),
            None,
        );
        let product = dag.add_node(
            RiscOp::Mul,
            vec![one_hot_exp, values_exp],
            t3(4, 2, 3),
            None,
        );
        let gathered = dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![product],
            mat(4, 3),
            None,
        );
        dag.add_root(gathered);

        let specialized = specialize_for_blas(&dag);
        assert!(specialized.nodes().iter().any(|node| {
            matches!(node.op, RiscOp::Gather { axis: 0 })
                && node.output_type.dims == vec![DimInfo::Lit(4), DimInfo::Lit(3)]
        }));
        assert!(
            !specialized
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::OneHot { .. } | RiscOp::BlasMatmul { .. })),
            "dense one-hot gather must become Gather before matmul recognition"
        );
        assert!(
            crate::verify::verify(&specialized).is_empty(),
            "specialized gather DAG must verify"
        );

        let inputs = std::collections::HashMap::from([
            (
                "values".to_string(),
                crate::eval::TensorValue::from_vec(
                    vec![2, 3],
                    vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
                ),
            ),
            (
                "indices".to_string(),
                crate::eval::TensorValue::from_vec(vec![4], vec![0.0, 1.0, 0.0, 1.0]),
            ),
        ]);
        let before = crate::eval::eval_tensor(&dag, &inputs).expect("dense gather eval");
        let after = crate::eval::eval_tensor(&specialized, &inputs).expect("specialized eval");
        assert_eq!(
            before[&gathered].data,
            after[specialized.roots().first().expect("root")].data
        );
    }

    #[test]
    fn one_hot_similar_tree_with_wrong_reduction_axis_does_not_false_match_gather() {
        let mut dag = Dag::new();
        let values = dag.add_node(
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            mat(2, 3),
            None,
        );
        let indices = dag.add_node(
            RiscOp::Load {
                name: "indices".into(),
            },
            vec![],
            vec_i32(4),
            None,
        );
        let one_hot = dag.add_node(RiscOp::OneHot { vocab: 2 }, vec![indices], mat(4, 2), None);
        let one_hot_exp = dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: DimExpr::Concrete(3),
            },
            vec![one_hot],
            t3(4, 2, 3),
            None,
        );
        let values_exp = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::Concrete(4),
            },
            vec![values],
            t3(4, 2, 3),
            None,
        );
        let product = dag.add_node(
            RiscOp::Mul,
            vec![one_hot_exp, values_exp],
            t3(4, 2, 3),
            None,
        );
        let not_gather = dag.add_node(
            RiscOp::Sum {
                axis: 2,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![product],
            mat(4, 2),
            None,
        );
        dag.add_root(not_gather);

        let specialized = specialize_for_blas(&dag);
        assert!(
            !specialized
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Gather { .. })),
            "the dense gather recognizer must reject similar one-hot trees \
             that reduce a non-vocabulary axis"
        );
        assert!(
            !specialized
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::OneHot { .. })),
            "the non-matching internal OneHot should still lower before backend codegen"
        );
        assert!(
            crate::verify::verify(&specialized).is_empty(),
            "fallback-lowered non-gather DAG must verify"
        );
    }

    #[test]
    fn unmatched_one_hot_lowers_to_primitive_ir() {
        let mut dag = Dag::new();
        let indices = dag.add_node(
            RiscOp::Load {
                name: "indices".into(),
            },
            vec![],
            vec_i32(3),
            None,
        );
        let one_hot = dag.add_node(
            RiscOp::OneHot { vocab: 3 },
            vec![indices],
            TensorType {
                dims: vec![DimInfo::Lit(3), DimInfo::Lit(3)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(one_hot);

        let specialized = specialize_for_blas(&dag);
        assert!(
            !specialized
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::OneHot { .. })),
            "fallback lowering must remove unmatched OneHot"
        );
        assert!(
            crate::verify::verify(&specialized).is_empty(),
            "lowered one_hot DAG must verify"
        );

        let inputs = std::collections::HashMap::from([(
            "indices".to_string(),
            crate::eval::TensorValue::from_vec(vec![3], vec![2.0, 0.0, 1.0]),
        )]);
        let before = crate::eval::eval_tensor(&dag, &inputs).expect("one_hot eval");
        let after = crate::eval::eval_tensor(&specialized, &inputs).expect("lowered eval");
        assert_eq!(
            before[&one_hot].data,
            after[specialized.roots().first().expect("root")].data
        );
        assert_eq!(
            after[specialized.roots().first().expect("root")].data,
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
    }
}
