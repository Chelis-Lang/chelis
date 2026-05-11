//! DAG structural verification.

use crate::dag::{Dag, DimInfo, RiscOp};
#[allow(unused_imports)]
use chelis_types::types::Prim;

/// Verify structural invariants of the DAG. Returns a list of error messages (empty = valid).
// `collapsible_match` (rust 1.95+) flags `match { X => { if cond { ... } } }` patterns.
// The validation arms here have large bodies and no else branch, so converting to match
// guards would require dedenting ~90 lines per arm with no readability win; the `if`
// inside an otherwise-empty arm body is the clearer expression of intent.
#[allow(clippy::collapsible_match)]
pub fn verify(dag: &Dag) -> Vec<String> {
    let mut errors = Vec::new();
    let mut consumers = vec![0usize; dag.len()];
    let mut load_types = std::collections::HashMap::<String, crate::dag::TensorType>::new();
    for node in dag.nodes() {
        for &input_id in &node.inputs {
            if input_id.0 < consumers.len() {
                consumers[input_id.0] += 1;
            }
        }

        if let RiscOp::Load { name } = &node.op {
            if let Some(prev_ty) = load_types.get(name.as_str()) {
                if prev_ty != &node.output_type {
                    errors.push(format!(
                        "load '{}' has inconsistent tensor types: {:?} vs {:?}",
                        name, prev_ty, node.output_type
                    ));
                }
            } else {
                load_types.insert(name.as_str().to_string(), node.output_type.clone());
            }
        }
    }

    for node in dag.nodes() {
        // Check that inputs reference valid, earlier nodes.
        for &input_id in &node.inputs {
            if input_id.0 >= node.id.0 {
                errors.push(format!(
                    "node {} references non-earlier node {}",
                    node.id.0, input_id.0
                ));
            }
            if dag.get(input_id).is_none() {
                errors.push(format!(
                    "node {} references nonexistent node {}",
                    node.id.0, input_id.0
                ));
            }
        }

        // Check arity.
        let arity = node.inputs.len();
        match &node.op {
            RiscOp::Add | RiscOp::Mul | RiscOp::CmpLt | RiscOp::MaxElem => {
                if arity != 2 {
                    errors.push(format!(
                        "binary op at node {} has {} inputs (expected 2)",
                        node.id.0, arity
                    ));
                }

                // C1: precision consistency check for binary ops.
                if arity == 2
                    && let (Some(lhs), Some(rhs)) =
                        (dag.get(node.inputs[0]), dag.get(node.inputs[1]))
                {
                    if lhs.output_type.precision != rhs.output_type.precision {
                        errors.push(format!(
                            "binary op at node {} has mismatched precisions: {:?} vs {:?}",
                            node.id.0, lhs.output_type.precision, rhs.output_type.precision
                        ));
                    }

                    // C2: dimension matching for binary ops.
                    let l_dims = &lhs.output_type.dims;
                    let r_dims = &rhs.output_type.dims;
                    if l_dims.len() != r_dims.len() {
                        errors.push(format!(
                            "binary op at node {} has mismatched dimension count: {} vs {}",
                            node.id.0,
                            l_dims.len(),
                            r_dims.len()
                        ));
                    } else {
                        for (i, (ld, rd)) in l_dims.iter().zip(r_dims.iter()).enumerate() {
                            if !dims_compatible(ld, rd) {
                                errors.push(format!(
                                    "binary op at node {} has mismatched dimension at axis {}: {:?} vs {:?}",
                                    node.id.0, i, ld, rd
                                ));
                            }
                        }
                    }
                }
            }
            RiscOp::BlasMatmul {
                batch_dims,
                m,
                n,
                k,
                ..
            } => {
                if arity != 2 {
                    errors.push(format!(
                        "blas matmul at node {} has {} inputs (expected 2)",
                        node.id.0, arity
                    ));
                }
                if arity == 2
                    && let (Some(lhs), Some(rhs)) =
                        (dag.get(node.inputs[0]), dag.get(node.inputs[1]))
                {
                    if lhs.output_type.precision != rhs.output_type.precision {
                        errors.push(format!(
                            "blas matmul at node {} has mismatched precisions: {:?} vs {:?}",
                            node.id.0, lhs.output_type.precision, rhs.output_type.precision
                        ));
                    }
                    if lhs.output_type.dims.len() < 2 || rhs.output_type.dims.len() < 2 {
                        errors.push(format!(
                            "blas matmul at node {} expects rank >= 2 inputs, got rank {} and {}",
                            node.id.0,
                            lhs.output_type.dims.len(),
                            rhs.output_type.dims.len()
                        ));
                    }
                    if node.output_type.dims.len() < 2 {
                        errors.push(format!(
                            "blas matmul at node {} expects rank >= 2 output, got rank {}",
                            node.id.0,
                            node.output_type.dims.len()
                        ));
                    }
                    if m.as_concrete() == Some(0)
                        || n.as_concrete() == Some(0)
                        || k.as_concrete() == Some(0)
                    {
                        errors.push(format!(
                            "blas matmul at node {} has zero dimension m={m} n={n} k={k}",
                            node.id.0
                        ));
                    }
                    if node.output_type.dims.len() >= 2 {
                        let out_batch_len = node.output_type.dims.len() - 2;
                        if batch_dims.len() != out_batch_len {
                            errors.push(format!(
                                "blas matmul at node {} has {} batch dims for rank {} output",
                                node.id.0,
                                batch_dims.len(),
                                node.output_type.dims.len()
                            ));
                        }
                        let expected_out_dims = batch_dims
                            .iter()
                            .cloned()
                            .chain([m.clone(), n.clone()])
                            .collect::<Vec<_>>();
                        let actual_out_dims = node
                            .output_type
                            .dims
                            .iter()
                            .map(crate::dag::DimExpr::from)
                            .collect::<Vec<_>>();
                        if expected_out_dims != actual_out_dims {
                            errors.push(format!(
                                "blas matmul at node {} has output dims {:?}, expected {:?}",
                                node.id.0, actual_out_dims, expected_out_dims
                            ));
                        }
                    }
                    if lhs.output_type.dims.len() >= 2 && rhs.output_type.dims.len() >= 2 {
                        let lhs_dims = lhs
                            .output_type
                            .dims
                            .iter()
                            .map(crate::dag::DimExpr::from)
                            .collect::<Vec<_>>();
                        let rhs_dims = rhs
                            .output_type
                            .dims
                            .iter()
                            .map(crate::dag::DimExpr::from)
                            .collect::<Vec<_>>();
                        let lhs_matrix = &lhs_dims[lhs_dims.len() - 2..];
                        let rhs_matrix = &rhs_dims[rhs_dims.len() - 2..];
                        if lhs_matrix != [m.clone(), k.clone()]
                            || rhs_matrix != [k.clone(), n.clone()]
                        {
                            errors.push(format!(
                                "blas matmul at node {} has incompatible matrix dims",
                                node.id.0
                            ));
                        }
                    }
                }
            }
            RiscOp::Gather { axis } => {
                if arity != 2 {
                    errors.push(format!(
                        "gather at node {} has {} inputs (expected 2)",
                        node.id.0, arity
                    ));
                }
                if arity == 2
                    && let (Some(values), Some(indices)) =
                        (dag.get(node.inputs[0]), dag.get(node.inputs[1]))
                {
                    if !matches!(indices.output_type.precision, Prim::Int32 | Prim::Int64) {
                        errors.push(format!(
                            "gather at node {} requires int32/int64 indices, got {:?}",
                            node.id.0, indices.output_type.precision
                        ));
                    }
                    if node.output_type.precision != values.output_type.precision {
                        errors.push(format!(
                            "gather at node {} output precision {:?} must match values precision {:?}",
                            node.id.0,
                            node.output_type.precision,
                            values.output_type.precision
                        ));
                    }
                    if *axis >= values.output_type.dims.len() {
                        errors.push(format!(
                            "gather at node {} has axis {} out of bounds for rank {}",
                            node.id.0,
                            axis,
                            values.output_type.dims.len()
                        ));
                    } else {
                        let mut expected = Vec::new();
                        expected.extend_from_slice(&values.output_type.dims[..*axis]);
                        expected.extend(indices.output_type.dims.iter().cloned());
                        expected.extend_from_slice(&values.output_type.dims[*axis + 1..]);
                        if node.output_type.dims != expected {
                            errors.push(format!(
                                "gather at node {} has output dims {:?}, expected {:?}",
                                node.id.0, node.output_type.dims, expected
                            ));
                        }
                    }
                }
            }
            RiscOp::ScatterAdd { axis } => {
                verify_scatter_like(node, dag, *axis, "scatter_add", &mut errors);
            }
            RiscOp::Scatter { axis } => {
                // Replace-scatter (last-write-wins) shares the input
                // arity / index-precision / shape contract with
                // scatter_add — only the duplicate-index semantics
                // differ (replace vs accumulate), which is a runtime
                // concern not a structural one.
                verify_scatter_like(node, dag, *axis, "scatter_replace", &mut errors);
            }
            RiscOp::Neg
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
            | RiscOp::Realize
            | RiscOp::Sum { .. }
            | RiscOp::MaxReduce { .. }
            | RiscOp::MinReduce { .. }
            | RiscOp::ProdReduce { .. }
            | RiscOp::Argmax { .. }
            | RiscOp::Argmin { .. }
            | RiscOp::Reshape { .. }
            | RiscOp::Permute { .. }
            | RiscOp::Expand { .. }
            | RiscOp::OneHot { .. }
            | RiscOp::Pad { .. }
            | RiscOp::Shrink { .. }
            | RiscOp::Stride { .. }
            | RiscOp::Cast { .. } => {
                if arity != 1 {
                    errors.push(format!(
                        "unary op at node {} has {} inputs (expected 1)",
                        node.id.0, arity
                    ));
                }
            }
            RiscOp::Store { .. } => {
                if arity != 1 {
                    errors.push(format!(
                        "store op at node {} has {} inputs (expected 1)",
                        node.id.0, arity
                    ));
                }
            }
            RiscOp::Const { .. } | RiscOp::Load { .. } => {
                if arity != 0 {
                    errors.push(format!(
                        "memory op at node {} has {} inputs (expected 0)",
                        node.id.0, arity
                    ));
                }
            }
            RiscOp::FusedElem { ops } => {
                if ops.is_empty() {
                    errors.push(format!("fused elem at node {} has no steps", node.id.0));
                }
            }
        }

        // C3: validate reduction axis bounds.
        match &node.op {
            RiscOp::Sum { axis, .. }
            | RiscOp::MaxReduce { axis }
            | RiscOp::MinReduce { axis }
            | RiscOp::ProdReduce { axis }
            | RiscOp::Argmax { axis }
            | RiscOp::Argmin { axis } => {
                if arity == 1
                    && let Some(input) = dag.get(node.inputs[0])
                {
                    let ndims = input.output_type.dims.len();
                    if ndims == 0 || *axis >= ndims {
                        errors.push(format!(
                            "reduction op at node {} has axis {} but input has {} dimensions",
                            node.id.0, axis, ndims
                        ));
                    }
                }
            }
            _ => {}
        }

        // C3a (WS-A0): per spec/04-type-system.md §5.7.1 the result
        // precision of `reduce_sum` IS the accumulator precision; the
        // IR invariant is `Sum.output_type.precision == accumulator`.
        // For BlasMatmul, the accumulator must be at least as wide as
        // the operand precision and at least as wide as the spec
        // default for that operand precision.
        match &node.op {
            RiscOp::Sum { accumulator, .. } => {
                if node.output_type.precision != *accumulator {
                    errors.push(format!(
                        "reduce_sum at node {} has output precision `{}` but \
                         accumulator `{}`; per spec/04-type-system.md §5.7.1 \
                         the result precision must equal the accumulator",
                        node.id.0,
                        node.output_type.precision.name(),
                        accumulator.name()
                    ));
                }
            }
            RiscOp::BlasMatmul { accumulator, .. } => {
                if arity == 2
                    && let Some(lhs) = dag.get(node.inputs[0])
                {
                    let operand = lhs.output_type.precision;
                    match RiscOp::default_matmul_accumulator(operand) {
                        Ok(default) => {
                            if !crate::dag::accumulator_at_least_as_wide(
                                operand,
                                *accumulator,
                                default,
                            ) {
                                errors.push(format!(
                                    "matmul at node {} has accumulator `{}` narrower than \
                                     the spec/04-type-system.md §5.7.1 default `{}` for \
                                     operand precision `{}`",
                                    node.id.0,
                                    accumulator.name(),
                                    default.name(),
                                    operand.name(),
                                ));
                            }
                        }
                        Err(msg) => errors.push(format!("matmul at node {}: {msg}", node.id.0)),
                    }
                }
            }
            _ => {}
        }

        // C4: transcendental ops require float precision.
        match &node.op {
            RiscOp::Exp
            | RiscOp::Log
            | RiscOp::Sin
            | RiscOp::Sqrt
            | RiscOp::Cos
            | RiscOp::Tan
            | RiscOp::Atan
            | RiscOp::Abs
            | RiscOp::Floor
            | RiscOp::Ceil => {
                if arity == 1
                    && let Some(input) = dag.get(node.inputs[0])
                    && !input.output_type.precision.is_float()
                {
                    errors.push(format!(
                        "transcendental op {:?} at node {} requires float input, got {:?}",
                        node.op, node.id.0, input.output_type.precision
                    ));
                }
            }
            _ => {}
        }

        // C5: CmpLt output must be Bool.
        if matches!(&node.op, RiscOp::CmpLt) && node.output_type.precision != Prim::Bool {
            errors.push(format!(
                "cmplt at node {} has output precision {:?}, expected Bool",
                node.id.0, node.output_type.precision
            ));
        }

        // C6: Permute validation.
        if let RiscOp::Permute { axes } = &node.op
            && arity == 1
        {
            let input = dag.get(node.inputs[0]).unwrap();
            let rank = input.output_type.dims.len();
            if axes.len() != rank {
                errors.push(format!(
                    "permute at node {}: axes len {} != input rank {}",
                    node.id.0,
                    axes.len(),
                    rank
                ));
            }
            let mut seen = vec![false; rank];
            for &a in axes {
                if a >= rank {
                    errors.push(format!(
                        "permute at node {}: axis {} >= rank {}",
                        node.id.0, a, rank
                    ));
                } else if seen[a] {
                    errors.push(format!(
                        "permute at node {}: duplicate axis {}",
                        node.id.0, a
                    ));
                } else {
                    seen[a] = true;
                }
            }
        }

        // C7: Reshape validation — product of dims must match.
        if let RiscOp::Reshape { new_shape } = &node.op
            && arity == 1
        {
            let input = dag.get(node.inputs[0]).unwrap();
            let old_product = dim_product(&input.output_type.dims);
            let new_product = dim_product(new_shape);
            if let (Some(old), Some(new)) = (old_product, new_product)
                && old != new
            {
                errors.push(format!(
                    "reshape at node {}: product mismatch {} vs {}",
                    node.id.0, old, new
                ));
            }
        }

        // C8: Cast validation — dims must not change, output precision must match target.
        if let RiscOp::Cast { new_precision } = &node.op
            && arity == 1
        {
            let input = dag.get(node.inputs[0]).unwrap();
            if input.output_type.dims != node.output_type.dims {
                errors.push(format!("cast at node {}: dims changed", node.id.0));
            }
            if node.output_type.precision != *new_precision {
                errors.push(format!(
                    "cast at node {}: output precision doesn't match cast target",
                    node.id.0
                ));
            }
        }

        // C9: Reduction output rank check.
        match &node.op {
            RiscOp::Sum { .. }
            | RiscOp::MaxReduce { .. }
            | RiscOp::MinReduce { .. }
            | RiscOp::ProdReduce { .. }
            | RiscOp::Argmax { .. }
            | RiscOp::Argmin { .. } => {
                if arity == 1 {
                    let input = dag.get(node.inputs[0]).unwrap();
                    let expected_rank = input.output_type.dims.len().saturating_sub(1);
                    if node.output_type.dims.len() != expected_rank {
                        errors.push(format!(
                            "reduction at node {}: output rank {} != expected {}",
                            node.id.0,
                            node.output_type.dims.len(),
                            expected_rank
                        ));
                    }
                }
            }
            _ => {}
        }

        // C10: Movement op shape validation.
        match &node.op {
            RiscOp::Expand { axis, size } => {
                if arity == 1 {
                    let input = dag.get(node.inputs[0]).unwrap();
                    let input_rank = input.output_type.dims.len();
                    let output_rank = node.output_type.dims.len();
                    if *axis > input_rank {
                        errors.push(format!(
                            "expand at node {}: axis {} > input rank {}",
                            node.id.0, axis, input_rank
                        ));
                    }
                    if matches!(size.as_concrete(), Some(0)) {
                        errors.push(format!("expand at node {}: size must be > 0", node.id.0));
                    }
                    if node.output_type.precision != input.output_type.precision {
                        errors.push(format!(
                            "expand at node {}: output precision {:?} != input precision {:?}",
                            node.id.0, node.output_type.precision, input.output_type.precision
                        ));
                    }
                    if output_rank == input_rank + 1 && *axis <= input_rank {
                        for out_i in 0..output_rank {
                            if out_i == *axis {
                                if let Some(out_size) =
                                    dim_known_size(&node.output_type.dims[out_i])
                                    && size.as_concrete().is_some_and(|size| out_size != size)
                                {
                                    errors.push(format!(
                                        "expand at node {}: inserted axis {} has size {}, expected {}",
                                        node.id.0, axis, out_size, size
                                    ));
                                }
                            } else {
                                let in_i = if out_i < *axis { out_i } else { out_i - 1 };
                                if !dims_compatible(
                                    &node.output_type.dims[out_i],
                                    &input.output_type.dims[in_i],
                                ) {
                                    errors.push(format!(
                                        "expand at node {}: output axis {} {:?} incompatible with input axis {} {:?}",
                                        node.id.0, out_i, node.output_type.dims[out_i], in_i, input.output_type.dims[in_i]
                                    ));
                                }
                            }
                        }
                    } else if output_rank == input_rank {
                        if *axis >= input_rank {
                            errors.push(format!(
                                "expand at node {}: axis {} >= input rank {} for same-rank expand",
                                node.id.0, axis, input_rank
                            ));
                        } else {
                            if let Some(in_size) = dim_known_size(&input.output_type.dims[*axis])
                                && in_size != 1
                            {
                                errors.push(format!(
                                    "expand at node {}: same-rank expand requires input axis {} to have size 1, got {}",
                                    node.id.0, axis, in_size
                                ));
                            }
                            if let Some(out_size) = dim_known_size(&node.output_type.dims[*axis])
                                && size.as_concrete().is_some_and(|size| out_size != size)
                            {
                                errors.push(format!(
                                    "expand at node {}: output axis {} has size {}, expected {}",
                                    node.id.0, axis, out_size, size
                                ));
                            }
                            for out_i in 0..output_rank {
                                if out_i == *axis {
                                    continue;
                                }
                                if !dims_compatible(
                                    &node.output_type.dims[out_i],
                                    &input.output_type.dims[out_i],
                                ) {
                                    errors.push(format!(
                                        "expand at node {}: output axis {} {:?} incompatible with input axis {} {:?}",
                                        node.id.0, out_i, node.output_type.dims[out_i], out_i, input.output_type.dims[out_i]
                                    ));
                                }
                            }
                        }
                    } else {
                        errors.push(format!(
                            "expand at node {}: output rank {} must equal input rank {} or {}",
                            node.id.0,
                            output_rank,
                            input_rank,
                            input_rank + 1
                        ));
                    }
                }
            }
            RiscOp::OneHot { vocab } => {
                if arity == 1 {
                    let input = dag.get(node.inputs[0]).unwrap();
                    if *vocab == 0 {
                        errors.push(format!("one_hot at node {}: vocab must be > 0", node.id.0));
                    }
                    if !matches!(input.output_type.precision, Prim::Int32 | Prim::Int64) {
                        errors.push(format!(
                            "one_hot at node {} requires int32/int64 indices, got {:?}",
                            node.id.0, input.output_type.precision
                        ));
                    }
                    if node.output_type.precision != Prim::F32 {
                        errors.push(format!(
                            "one_hot at node {} output precision {:?}, expected F32",
                            node.id.0, node.output_type.precision
                        ));
                    }
                    let mut expected = input.output_type.dims.clone();
                    expected.push(DimInfo::Lit(*vocab));
                    if node.output_type.dims != expected {
                        errors.push(format!(
                            "one_hot at node {} has output dims {:?}, expected {:?}",
                            node.id.0, node.output_type.dims, expected
                        ));
                    }
                }
            }
            RiscOp::Pad { padding, .. } => {
                if arity == 1 {
                    let input = dag.get(node.inputs[0]).unwrap();
                    let input_rank = input.output_type.dims.len();
                    if padding.len() != input_rank {
                        errors.push(format!(
                            "pad at node {}: padding len {} != input rank {}",
                            node.id.0,
                            padding.len(),
                            input_rank
                        ));
                    }
                    if node.output_type.precision != input.output_type.precision {
                        errors.push(format!(
                            "pad at node {}: output precision {:?} != input precision {:?}",
                            node.id.0, node.output_type.precision, input.output_type.precision
                        ));
                    }
                    if node.output_type.dims.len() != input_rank {
                        errors.push(format!(
                            "pad at node {}: output rank {} != input rank {}",
                            node.id.0,
                            node.output_type.dims.len(),
                            input_rank
                        ));
                    } else {
                        for (axis, ((before, after), in_dim)) in padding
                            .iter()
                            .zip(input.output_type.dims.iter())
                            .enumerate()
                        {
                            if let Some(in_size) = dim_known_size(in_dim) {
                                let expected = in_size + before + after;
                                if let Some(out_size) = dim_known_size(&node.output_type.dims[axis])
                                    && out_size != expected
                                {
                                    errors.push(format!(
                                        "pad at node {}: output axis {} has size {}, expected {}",
                                        node.id.0, axis, out_size, expected
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            RiscOp::Shrink { bounds } => {
                if arity == 1 {
                    let input = dag.get(node.inputs[0]).unwrap();
                    let input_rank = input.output_type.dims.len();
                    if bounds.len() != input_rank {
                        errors.push(format!(
                            "shrink at node {}: bounds len {} != input rank {}",
                            node.id.0,
                            bounds.len(),
                            input_rank
                        ));
                    }
                    if node.output_type.precision != input.output_type.precision {
                        errors.push(format!(
                            "shrink at node {}: output precision {:?} != input precision {:?}",
                            node.id.0, node.output_type.precision, input.output_type.precision
                        ));
                    }
                    if node.output_type.dims.len() != input_rank {
                        errors.push(format!(
                            "shrink at node {}: output rank {} != input rank {}",
                            node.id.0,
                            node.output_type.dims.len(),
                            input_rank
                        ));
                    } else {
                        for (axis, ((start, end), in_dim)) in
                            bounds.iter().zip(input.output_type.dims.iter()).enumerate()
                        {
                            if start > end {
                                errors.push(format!(
                                    "shrink at node {}: axis {} has invalid bounds ({}, {})",
                                    node.id.0, axis, start, end
                                ));
                                continue;
                            }
                            if let Some(in_size) = dim_known_size(in_dim)
                                && *end > in_size
                            {
                                errors.push(format!(
                                    "shrink at node {}: axis {} end {} > input size {}",
                                    node.id.0, axis, end, in_size
                                ));
                            }
                            let expected = end - start;
                            if let Some(out_size) = dim_known_size(&node.output_type.dims[axis])
                                && out_size != expected
                            {
                                errors.push(format!(
                                    "shrink at node {}: output axis {} has size {}, expected {}",
                                    node.id.0, axis, out_size, expected
                                ));
                            }
                        }
                    }
                }
            }
            RiscOp::Stride { strides } => {
                if arity == 1 {
                    let input = dag.get(node.inputs[0]).unwrap();
                    let input_rank = input.output_type.dims.len();
                    if strides.len() != input_rank {
                        errors.push(format!(
                            "stride at node {}: strides len {} != input rank {}",
                            node.id.0,
                            strides.len(),
                            input_rank
                        ));
                    }
                    if node.output_type.precision != input.output_type.precision {
                        errors.push(format!(
                            "stride at node {}: output precision {:?} != input precision {:?}",
                            node.id.0, node.output_type.precision, input.output_type.precision
                        ));
                    }
                    if node.output_type.dims.len() != input_rank {
                        errors.push(format!(
                            "stride at node {}: output rank {} != input rank {}",
                            node.id.0,
                            node.output_type.dims.len(),
                            input_rank
                        ));
                    } else {
                        for (axis, (step, in_dim)) in strides
                            .iter()
                            .zip(input.output_type.dims.iter())
                            .enumerate()
                        {
                            if *step == 0 {
                                errors.push(format!(
                                    "stride at node {}: axis {} has invalid step 0",
                                    node.id.0, axis
                                ));
                                continue;
                            }
                            if let Some(in_size) = dim_known_size(in_dim) {
                                let expected = in_size.div_ceil(*step);
                                if let Some(out_size) = dim_known_size(&node.output_type.dims[axis])
                                    && out_size != expected
                                {
                                    errors.push(format!(
                                        "stride at node {}: output axis {} has size {}, expected {}",
                                        node.id.0, axis, out_size, expected
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        // C11: Store arity (already checked above, but explicit message).
        if let RiscOp::Store { .. } = &node.op
            && node.inputs.len() != 1
        {
            errors.push(format!(
                "store at node {} has {} inputs (expected 1)",
                node.id.0,
                node.inputs.len()
            ));
        }

        if matches!(node.op, RiscOp::Drop) {
            if dag.is_root(node.id) {
                errors.push(format!(
                    "drop at node {} is terminal and cannot be a DAG root",
                    node.id.0
                ));
            }
            if consumers[node.id.0] != 0 {
                errors.push(format!(
                    "drop at node {} is terminal and cannot be consumed",
                    node.id.0
                ));
            }
        }

        let is_implicit_root = dag.roots().is_empty() && node.id.0 + 1 == dag.len();
        if !dag.is_root(node.id)
            && !is_implicit_root
            && consumers[node.id.0] == 0
            && !matches!(node.op, RiscOp::Store { .. } | RiscOp::Drop)
        {
            errors.push(format!(
                "node {} is dangling: it has no consumers and is not a DAG root",
                node.id.0
            ));
        }
    }
    errors
}

/// Shared structural verification for replace-scatter and scatter-add.
/// Inputs are `[target, indices, updates]`; output shape equals
/// `target`; `updates` shape equals `target.dims[..axis] +
/// indices.dims + target.dims[axis+1..]`. Indices must be int32/int64.
fn verify_scatter_like(
    node: &crate::dag::DagNode,
    dag: &Dag,
    axis: usize,
    label: &str,
    errors: &mut Vec<String>,
) {
    let arity = node.inputs.len();
    if arity != 3 {
        errors.push(format!(
            "{label} at node {} has {} inputs (expected 3)",
            node.id.0, arity
        ));
        return;
    }
    if let (Some(target), Some(indices), Some(updates)) = (
        dag.get(node.inputs[0]),
        dag.get(node.inputs[1]),
        dag.get(node.inputs[2]),
    ) {
        if !matches!(indices.output_type.precision, Prim::Int32 | Prim::Int64) {
            errors.push(format!(
                "{label} at node {} requires int32/int64 indices, got {:?}",
                node.id.0, indices.output_type.precision
            ));
        }
        if updates.output_type.precision != target.output_type.precision {
            errors.push(format!(
                "{label} at node {} update precision {:?} must match target precision {:?}",
                node.id.0, updates.output_type.precision, target.output_type.precision
            ));
        }
        if node.output_type != target.output_type {
            errors.push(format!(
                "{label} at node {} output type must match target",
                node.id.0
            ));
        }
        if axis >= target.output_type.dims.len() {
            errors.push(format!(
                "{label} at node {} has axis {} out of bounds for rank {}",
                node.id.0,
                axis,
                target.output_type.dims.len()
            ));
        } else {
            let mut expected_updates = Vec::new();
            expected_updates.extend_from_slice(&target.output_type.dims[..axis]);
            expected_updates.extend(indices.output_type.dims.iter().cloned());
            expected_updates.extend_from_slice(&target.output_type.dims[axis + 1..]);
            if updates.output_type.dims != expected_updates {
                errors.push(format!(
                    "{label} at node {} has update dims {:?}, expected {:?}",
                    node.id.0, updates.output_type.dims, expected_updates
                ));
            }
        }
    }
}

/// Compute the product of known dimension sizes. Returns None if any dim is unknown.
fn dim_product(dims: &[DimInfo]) -> Option<usize> {
    let mut product = 1usize;
    for d in dims {
        match d {
            DimInfo::Lit(n) => product *= n,
            DimInfo::Named(_, Some(n)) => product *= n,
            DimInfo::Named(_, None) => return None,
        }
    }
    Some(product)
}

fn dim_known_size(dim: &DimInfo) -> Option<usize> {
    match dim {
        DimInfo::Lit(n) => Some(*n),
        DimInfo::Named(_, Some(n)) => Some(*n),
        DimInfo::Named(_, None) => None,
    }
}

/// Check if two dimension descriptors are compatible.
fn dims_compatible(a: &DimInfo, b: &DimInfo) -> bool {
    match (a, b) {
        (DimInfo::Lit(x), DimInfo::Lit(y)) => x == y,
        (DimInfo::Named(n1, s1), DimInfo::Named(n2, s2)) => {
            if n1 == n2 {
                // Same name: sizes must match if both known.
                match (s1, s2) {
                    (Some(a), Some(b)) => a == b,
                    _ => true,
                }
            } else {
                // Different names: compatible only if sizes match (if known).
                match (s1, s2) {
                    (Some(a), Some(b)) => a == b,
                    _ => true, // unknown sizes are assumed compatible
                }
            }
        }
        (DimInfo::Named(_, Some(s)), DimInfo::Lit(n))
        | (DimInfo::Lit(n), DimInfo::Named(_, Some(s))) => s == n,
        // Named with unknown size vs Lit: assume compatible (can't check).
        (DimInfo::Named(_, None), DimInfo::Lit(_)) | (DimInfo::Lit(_), DimInfo::Named(_, None)) => {
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{NodeId, RiscOp, TensorType};

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    fn tensor_ty(dims: &[usize], precision: Prim) -> TensorType {
        TensorType {
            dims: dims.iter().copied().map(DimInfo::Lit).collect(),
            precision,
        }
    }

    #[test]
    fn valid_dag_no_errors() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        assert!(verify(&dag).is_empty());
    }

    #[test]
    fn drop_root_is_rejected() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let drop = dag.add_node(RiscOp::Drop, vec![a], scalar_f32(), None);
        dag.add_root(drop);
        let errs = verify(&dag);
        assert!(
            errs.iter().any(|e| e.contains("cannot be a DAG root")),
            "{errs:?}"
        );
    }

    #[test]
    fn consumed_drop_is_rejected() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let drop = dag.add_node(RiscOp::Drop, vec![a], scalar_f32(), None);
        dag.add_node(RiscOp::Neg, vec![drop], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(
            errs.iter().any(|e| e.contains("cannot be consumed")),
            "{errs:?}"
        );
    }

    #[test]
    fn copy_and_drop_require_one_input() {
        let mut copy_dag = Dag::new();
        copy_dag.add_node(RiscOp::Copy, vec![], scalar_f32(), None);
        let copy_errs = verify(&copy_dag);
        assert!(
            copy_errs.iter().any(|e| e.contains("unary op")),
            "{copy_errs:?}"
        );

        let mut drop_dag = Dag::new();
        drop_dag.add_node(RiscOp::Drop, vec![], scalar_f32(), None);
        let drop_errs = verify(&drop_dag);
        assert!(
            drop_errs.iter().any(|e| e.contains("unary op")),
            "{drop_errs:?}"
        );
    }

    #[test]
    fn bad_input_reference() {
        let mut dag = Dag::new();
        // Manually create a node that references a future node (impossible via normal API,
        // but we can test via the replace_node backdoor or by constructing the scenario).
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        // Node 1 references itself (not earlier).
        dag.add_node(RiscOp::Neg, vec![NodeId(1)], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs.iter().any(|e| e.contains("non-earlier node")));
        let _ = a;
    }

    #[test]
    fn wrong_arity_binary() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        // Add with only 1 input.
        dag.add_node(RiscOp::Add, vec![a], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs[0].contains("binary op"));
    }

    #[test]
    fn wrong_arity_unary() {
        let mut dag = Dag::new();
        // Neg with 0 inputs.
        dag.add_node(RiscOp::Neg, vec![], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs[0].contains("unary op"));
    }

    #[test]
    fn wrong_arity_memory() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        // Const with an input (should have 0).
        dag.add_node(RiscOp::Const { value: 2.0 }, vec![a], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs[0].contains("memory op"));
    }

    // --- C1: precision consistency ---

    #[test]
    fn c1_mismatched_precision_binary_op() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b_ty = TensorType {
            dims: vec![],
            precision: Prim::F64,
        };
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], b_ty, None);
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs.iter().any(|e| e.contains("mismatched precisions")));
    }

    #[test]
    fn c1_matching_precision_ok() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Mul, vec![a, b], scalar_f32(), None);
        assert!(verify(&dag).is_empty());
    }

    // --- C2: dimension matching ---

    #[test]
    fn c2_mismatched_dim_count() {
        let mut dag = Dag::new();
        let ty1 = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let ty2 = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], ty1, None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], ty2, None);
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("mismatched dimension count"))
        );
    }

    #[test]
    fn c2_mismatched_dim_size() {
        let mut dag = Dag::new();
        let ty1 = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let ty2 = TensorType {
            dims: vec![DimInfo::Lit(5)],
            precision: Prim::F32,
        };
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], ty1, None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], ty2, None);
        dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("mismatched dimension at axis"))
        );
    }

    // --- C3: reduction axis bounds ---

    #[test]
    fn c3_sum_axis_out_of_bounds() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], ty.clone(), None);
        dag.add_node(
            RiscOp::Sum {
                axis: 5,
                accumulator: Prim::F32,
            },
            vec![x],
            scalar_f32(),
            None,
        );
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("axis 5")));
    }

    #[test]
    fn c3_max_reduce_on_scalar() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::MaxReduce { axis: 0 }, vec![x], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("axis 0 but input has 0 dimensions"))
        );
    }

    #[test]
    fn c3_sum_valid_axis() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let out_ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], ty, None);
        dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: Prim::F32,
            },
            vec![x],
            out_ty,
            None,
        );
        assert!(verify(&dag).is_empty());
    }

    // --- C4: transcendental float-only ---

    #[test]
    fn c4_exp_on_int_is_error() {
        let mut dag = Dag::new();
        let int_ty = TensorType {
            dims: vec![],
            precision: Prim::Int32,
        };
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], int_ty.clone(), None);
        dag.add_node(RiscOp::Exp, vec![x], int_ty, None);
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("transcendental")));
    }

    #[test]
    fn c4_sqrt_on_float_ok() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 4.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Sqrt, vec![x], scalar_f32(), None);
        assert!(verify(&dag).is_empty());
    }

    // --- C5: CmpLt output must be Bool ---

    #[test]
    fn c5_cmplt_non_bool_output_is_error() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        // Wrong: output is F32 instead of Bool.
        dag.add_node(RiscOp::CmpLt, vec![a, b], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("cmplt") && e.contains("Bool"))
        );
    }

    #[test]
    fn c5_cmplt_bool_output_ok() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        let bool_ty = TensorType {
            dims: vec![],
            precision: Prim::Bool,
        };
        dag.add_node(RiscOp::CmpLt, vec![a, b], bool_ty, None);
        assert!(verify(&dag).is_empty());
    }

    // --- C10: movement shape validation ---

    #[test]
    fn c10_expand_axis_out_of_bounds_is_error() {
        let mut dag = Dag::new();
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty, None);
        dag.add_node(
            RiscOp::Expand {
                axis: 2,
                size: crate::dag::DimExpr::Concrete(4),
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("expand")));
    }

    #[test]
    fn c10_expand_same_rank_broadcast_ok() {
        let mut dag = Dag::new();
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty, None);
        dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: crate::dag::DimExpr::Concrete(4),
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        assert!(verify(&dag).is_empty());
    }

    #[test]
    fn c10_expand_same_rank_requires_unit_input_axis() {
        let mut dag = Dag::new();
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty, None);
        dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: crate::dag::DimExpr::Concrete(4),
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("requires input axis 0 to have size 1"))
        );
    }

    #[test]
    fn c10_pad_wrong_rank_is_error() {
        let mut dag = Dag::new();
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty, None);
        dag.add_node(
            RiscOp::Pad {
                padding: vec![(1, 1)],
                fill: 0.0,
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(5), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("pad")));
    }

    #[test]
    fn c10_shrink_invalid_bounds_is_error() {
        let mut dag = Dag::new();
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(8)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty, None);
        dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(6, 2)],
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("shrink")));
    }

    #[test]
    fn c10_stride_zero_step_is_error() {
        let mut dag = Dag::new();
        let input_ty = TensorType {
            dims: vec![DimInfo::Lit(8)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty, None);
        dag.add_node(
            RiscOp::Stride { strides: vec![0] },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(8)],
                precision: Prim::F32,
            },
            None,
        );
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("stride")));
    }

    // --- C11: Store arity ---

    #[test]
    fn store_correct_arity() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        dag.add_node(
            RiscOp::Store { name: "out".into() },
            vec![x],
            scalar_f32(),
            None,
        );
        assert!(verify(&dag).is_empty());
    }

    #[test]
    fn store_wrong_arity_zero_inputs() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Store { name: "out".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs.iter().any(|e| e.contains("store")));
    }

    #[test]
    fn store_wrong_arity_two_inputs() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        dag.add_node(
            RiscOp::Store { name: "out".into() },
            vec![a, b],
            scalar_f32(),
            None,
        );
        let errs = verify(&dag);
        assert!(!errs.is_empty());
        assert!(errs.iter().any(|e| e.contains("store")));
    }

    #[test]
    fn dangling_nonfinal_node_is_error() {
        let mut dag = Dag::new();
        dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        let errs = verify(&dag);
        assert!(errs.iter().any(|e| e.contains("dangling")));
    }

    #[test]
    fn root_nodes_are_not_dangling() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
        dag.add_root(a);
        dag.add_root(b);
        assert!(verify(&dag).is_empty());
    }

    #[test]
    fn load_name_type_mismatch_is_error() {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(3)],
                precision: Prim::F32,
            },
            None,
        );
        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("load 'x' has inconsistent tensor types"))
        );
    }

    #[test]
    fn gather_requires_integer_indices_and_matching_output_precision() {
        let mut dag = Dag::new();
        let values = dag.add_node(
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F64),
            None,
        );
        let bad_indices = dag.add_node(
            RiscOp::Const { value: 0.0 },
            vec![],
            tensor_ty(&[3], Prim::F32),
            None,
        );
        dag.add_node(
            RiscOp::Gather { axis: 0 },
            vec![values, bad_indices],
            tensor_ty(&[3, 2], Prim::F32),
            None,
        );

        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("requires int32/int64 indices")),
            "expected integer-index diagnostic, got {errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("output precision")),
            "expected output-precision diagnostic, got {errs:?}"
        );
    }

    #[test]
    fn scatter_add_requires_integer_indices_and_matching_update_precision() {
        let mut dag = Dag::new();
        let target = dag.add_node(
            RiscOp::Load {
                name: "target".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F64),
            None,
        );
        let bad_indices = dag.add_node(
            RiscOp::Const { value: 0.0 },
            vec![],
            tensor_ty(&[3], Prim::F32),
            None,
        );
        let bad_updates = dag.add_node(
            RiscOp::Const { value: 1.0 },
            vec![],
            tensor_ty(&[3, 2], Prim::F32),
            None,
        );
        dag.add_node(
            RiscOp::ScatterAdd { axis: 0 },
            vec![target, bad_indices, bad_updates],
            tensor_ty(&[4, 2], Prim::F64),
            None,
        );

        let errs = verify(&dag);
        assert!(
            errs.iter()
                .any(|e| e.contains("requires int32/int64 indices")),
            "expected integer-index diagnostic, got {errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("update precision")),
            "expected update-precision diagnostic, got {errs:?}"
        );
    }
}
