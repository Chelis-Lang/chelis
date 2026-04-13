//! Tier 2 decomposition helpers.
//!
//! These functions decompose Tier 2 (derived) operations into Tier 1 RISC DAG nodes.

use chelis_types::types::Prim;

use crate::dag::{Dag, DimExpr, DimInfo, NodeId, RiscOp, TensorType};

/// `sub(a, b)` = `add(a, neg(b))`
pub fn lower_sub(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let neg_b = dag.add_node(RiscOp::Neg, vec![b], ty.clone());
    dag.add_node(RiscOp::Add, vec![a, neg_b], ty.clone())
}

/// `relu(x)` = `max_elem(x, const(0))`
pub fn lower_relu(dag: &mut Dag, x: NodeId, ty: &TensorType) -> NodeId {
    let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], ty.clone());
    dag.add_node(RiscOp::MaxElem, vec![x, zero], ty.clone())
}

/// `sigmoid(x)` = `1 / (1 + exp(-x))`
///
/// Lowered as: `exp(neg(log(add(const(1), exp(neg(x))))))`
pub fn lower_sigmoid(dag: &mut Dag, x: NodeId, ty: &TensorType) -> NodeId {
    let neg_x = dag.add_node(RiscOp::Neg, vec![x], ty.clone());
    let exp_neg = dag.add_node(RiscOp::Exp, vec![neg_x], ty.clone());
    let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], ty.clone());
    let sum = dag.add_node(RiscOp::Add, vec![one, exp_neg], ty.clone());
    // recip(sum) = exp(neg(log(sum)))
    let log_sum = dag.add_node(RiscOp::Log, vec![sum], ty.clone());
    let neg_log = dag.add_node(RiscOp::Neg, vec![log_sum], ty.clone());
    dag.add_node(RiscOp::Exp, vec![neg_log], ty.clone())
}

/// `div(a, b)` = `mul(a, recip(b))` where `recip(b) = exp(neg(log(b)))`
pub fn lower_div(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let log_b = dag.add_node(RiscOp::Log, vec![b], ty.clone());
    let neg_log = dag.add_node(RiscOp::Neg, vec![log_b], ty.clone());
    let recip_b = dag.add_node(RiscOp::Exp, vec![neg_log], ty.clone());
    dag.add_node(RiscOp::Mul, vec![a, recip_b], ty.clone())
}

/// H1: `gt(a, b)` = `cmplt(b, a)` (swap args)
pub fn lower_gt(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    dag.add_node(RiscOp::CmpLt, vec![b, a], bool_ty)
}

/// H1: `gte(a, b)` = `neg(cmplt(a, b))` — not (a < b)
/// Per spec §3.2: gte uses neg on bool (0/1 convention).
pub fn lower_gte(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    let lt = dag.add_node(RiscOp::CmpLt, vec![a, b], bool_ty.clone());
    // not(lt): cmplt(lt, const(1)) — if lt==0 then 0<1=true, if lt==1 then 1<1=false
    let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], bool_ty.clone());
    dag.add_node(RiscOp::CmpLt, vec![lt, one], bool_ty)
}

/// H1: `lte(a, b)` = `neg(cmplt(b, a))` — not (b < a)
pub fn lower_lte(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    let lt = dag.add_node(RiscOp::CmpLt, vec![b, a], bool_ty.clone());
    let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], bool_ty.clone());
    dag.add_node(RiscOp::CmpLt, vec![lt, one], bool_ty)
}

/// H1: `eq(a, b)` = not(or(cmplt(a,b), cmplt(b,a)))
/// `or` on bools = `max_elem`, `not` = `cmplt(x, const(1))`
pub fn lower_eq(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    let lt_ab = dag.add_node(RiscOp::CmpLt, vec![a, b], bool_ty.clone());
    let lt_ba = dag.add_node(RiscOp::CmpLt, vec![b, a], bool_ty.clone());
    let or = dag.add_node(RiscOp::MaxElem, vec![lt_ab, lt_ba], bool_ty.clone());
    let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], bool_ty.clone());
    dag.add_node(RiscOp::CmpLt, vec![or, one], bool_ty)
}

/// H1: `neq(a, b)` = `or(cmplt(a,b), cmplt(b,a))`
pub fn lower_neq(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    let lt_ab = dag.add_node(RiscOp::CmpLt, vec![a, b], bool_ty.clone());
    let lt_ba = dag.add_node(RiscOp::CmpLt, vec![b, a], bool_ty.clone());
    dag.add_node(RiscOp::MaxElem, vec![lt_ab, lt_ba], bool_ty)
}

/// H1: `min_elem(a, b)` = `neg(max_elem(neg(a), neg(b)))`
pub fn lower_min_elem(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let neg_a = dag.add_node(RiscOp::Neg, vec![a], ty.clone());
    let neg_b = dag.add_node(RiscOp::Neg, vec![b], ty.clone());
    let max = dag.add_node(RiscOp::MaxElem, vec![neg_a, neg_b], ty.clone());
    dag.add_node(RiscOp::Neg, vec![max], ty.clone())
}

/// H2: `and(a, b)` on bools = `mul(a, b)`
pub fn lower_and(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    dag.add_node(RiscOp::Mul, vec![a, b], bool_ty)
}

/// H2: `or(a, b)` on bools = `max_elem(a, b)`
pub fn lower_or(dag: &mut Dag, a: NodeId, b: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    dag.add_node(RiscOp::MaxElem, vec![a, b], bool_ty)
}

/// H2: `not(a)` on bools = `cmplt(a, const(1))` — flips 0->1, 1->0
pub fn lower_not(dag: &mut Dag, a: NodeId, ty: &TensorType) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    let one = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], bool_ty.clone());
    dag.add_node(RiscOp::CmpLt, vec![a, one], bool_ty)
}

// ---------------------------------------------------------------------------
// Helper functions for type manipulation
// ---------------------------------------------------------------------------

/// Produce a type with the given axis removed (for reductions).
pub fn reduced_type(ty: &TensorType, axis: usize) -> TensorType {
    let mut dims = ty.dims.clone();
    if axis < dims.len() {
        dims.remove(axis);
    }
    TensorType {
        dims,
        precision: ty.precision,
    }
}

/// Extract a concrete dimension size from a tensor type at the given axis.
pub fn dim_size(ty: &TensorType, axis: usize) -> Option<usize> {
    ty.dims.get(axis).and_then(|d| match d {
        DimInfo::Lit(n) => Some(*n),
        DimInfo::Named(_, Some(n)) => Some(*n),
        _ => None,
    })
}

/// Extract a runtime-capable dimension expression from a tensor type at the given axis.
pub fn dim_expr(ty: &TensorType, axis: usize) -> Option<DimExpr> {
    ty.dims.get(axis).map(DimExpr::from)
}

/// A scalar type with no dimensions and the given precision.
pub fn scalar_type(precision: Prim) -> TensorType {
    TensorType {
        dims: vec![],
        precision,
    }
}

fn require_dim(dim: Option<&DimInfo>, context: &str) -> DimInfo {
    dim.cloned().unwrap_or_else(|| {
        panic!("{context} requires a statically known axis in Phase 0e lowering")
    })
}

fn require_dim_extent(dim: &DimInfo, context: &str) -> usize {
    // Phase 0e IR lowering assumes dimensions are concrete by this point.
    // Named dimensions are for type-checking consistency; symbolic/runtime-sized
    // lowering is a Phase 1 IR/backend extension.
    match dim {
        DimInfo::Lit(n) => *n,
        DimInfo::Named(_, Some(n)) => *n,
        DimInfo::Named(name, None) => {
            panic!("{context} requires a concrete extent in Phase 0e lowering, got `{name}`")
        }
    }
}

fn require_axis_size(ty: &TensorType, axis: usize, context: &str) -> usize {
    dim_size(ty, axis).unwrap_or_else(|| {
        panic!("{context} requires a concrete extent for axis {axis} in Phase 0e lowering")
    })
}

fn expand_to_match(
    dag: &mut Dag,
    mut node: NodeId,
    mut ty: TensorType,
    target_dims: &[DimInfo],
) -> NodeId {
    while ty.dims.len() < target_dims.len() {
        let missing = target_dims.len() - ty.dims.len();
        let insert_at = 0;
        let source_dim = target_dims[missing - 1].clone();
        let size = DimExpr::from(&source_dim);
        let mut next_dims = ty.dims.clone();
        next_dims.insert(insert_at, source_dim);
        let next_ty = TensorType {
            dims: next_dims,
            precision: ty.precision,
        };
        node = dag.add_node(
            RiscOp::Expand {
                axis: insert_at,
                size,
            },
            vec![node],
            next_ty.clone(),
        );
        ty = next_ty;
    }
    node
}

// ---------------------------------------------------------------------------
// Tier 2 higher-level decompositions (spec §3.4, §4.1–4.2)
// ---------------------------------------------------------------------------

/// matmul(A: [i, j], B: [j, k]) -> [i, k]
///
/// Lowering (spec §4.1):
///   1. Expand A from [i, j] to [i, j, k] by adding a trailing dim
///   2. Expand B from [j, k] to [i, j, k] by adding a leading dim
///   3. Mul the expanded tensors -> [i, j, k]
///   4. Sum over axis 1 (the j dimension) -> [i, k]
///
pub fn lower_matmul(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    a_ty: &TensorType,
    b_ty: &TensorType,
) -> NodeId {
    // Extract dimension sizes: A is [i, j], B is [j, k].
    let i_dim = require_dim(a_ty.dims.first(), "matmul lhs axis 0");
    let j_dim = require_dim(
        a_ty.dims.get(1).or_else(|| b_ty.dims.first()),
        "matmul shared axis",
    );
    let k_dim = require_dim(b_ty.dims.get(1), "matmul rhs axis 1");
    let i_size = DimExpr::from(&i_dim);
    let k_size = DimExpr::from(&k_dim);

    // The intermediate expanded type is [i, j, k].
    let expanded_ty = TensorType {
        dims: vec![i_dim.clone(), j_dim.clone(), k_dim.clone()],
        precision: a_ty.precision,
    };

    // The result type is [i, k].
    let result_ty = TensorType {
        dims: vec![i_dim, k_dim],
        precision: a_ty.precision,
    };

    // 1. Expand A: add dim for k at axis 2 -> [i, j, k]
    let a_expanded = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: k_size,
        },
        vec![a],
        expanded_ty.clone(),
    );

    // 2. Expand B: add dim for i at axis 0 -> [i, j, k]
    let b_expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: i_size,
        },
        vec![b],
        expanded_ty.clone(),
    );

    // 3. Elementwise multiply -> [i, j, k]
    let product = dag.add_node(RiscOp::Mul, vec![a_expanded, b_expanded], expanded_ty);

    // 4. Sum over axis 1 (j) -> [i, k]
    dag.add_node(RiscOp::Sum { axis: 1 }, vec![product], result_ty)
}

/// softmax(x, axis) = exp(x - max_reduce(x, axis)) / sum(exp(x - max_reduce(x, axis)), axis)
///
/// Lowering (spec §4.2): numerically stable softmax via max subtraction.
pub fn lower_softmax(dag: &mut Dag, x: NodeId, axis: usize, ty: &TensorType) -> NodeId {
    let red_ty = reduced_type(ty, axis);
    let size = DimExpr::from(&require_dim(ty.dims.get(axis), "softmax axis"));

    // 1. max_reduce(x, axis)
    let max_val = dag.add_node(RiscOp::MaxReduce { axis }, vec![x], red_ty.clone());

    // 2. expand max back to original shape
    let max_expanded = dag.add_node(
        RiscOp::Expand {
            axis,
            size: size.clone(),
        },
        vec![max_val],
        ty.clone(),
    );

    // 3. shifted = x - max (numerical stability)
    let shifted = lower_sub(dag, x, max_expanded, ty);

    // 4. exp(shifted)
    let exp_shifted = dag.add_node(RiscOp::Exp, vec![shifted], ty.clone());

    // 5. sum(exp, axis)
    let sum_exp = dag.add_node(RiscOp::Sum { axis }, vec![exp_shifted], red_ty);

    // 6. expand sum back to original shape
    let sum_expanded = dag.add_node(RiscOp::Expand { axis, size }, vec![sum_exp], ty.clone());

    // 7. exp / sum
    lower_div(dag, exp_shifted, sum_expanded, ty)
}

/// mean(x, axis) = sum(x, axis) / dim_size
///
/// Lowering (spec §3.4): sum then divide by the axis size.
pub fn lower_mean(dag: &mut Dag, x: NodeId, axis: usize, ty: &TensorType) -> NodeId {
    let red_ty = reduced_type(ty, axis);
    let dim_size_val = require_axis_size(ty, axis, "mean") as f64;

    // sum(x, axis)
    let sum_node = dag.add_node(RiscOp::Sum { axis }, vec![x], red_ty.clone());

    // const(dim_size)
    let size_const = dag.add_node(
        RiscOp::Const {
            value: dim_size_val,
        },
        vec![],
        red_ty.clone(),
    );

    // sum / dim_size
    lower_div(dag, sum_node, size_const, &red_ty)
}

/// layer_norm(x, gamma, beta) over the last axis.
#[allow(clippy::too_many_arguments)]
pub fn lower_layer_norm(
    dag: &mut Dag,
    x: NodeId,
    gamma: NodeId,
    beta: NodeId,
    x_ty: &TensorType,
    gamma_ty: &TensorType,
    beta_ty: &TensorType,
    eps: f64,
) -> NodeId {
    let axis = x_ty.dims.len().saturating_sub(1);
    let axis_size = DimExpr::Concrete(require_axis_size(x_ty, axis, "layer_norm"));
    let mean = lower_mean(dag, x, axis, x_ty);
    let mean_expanded = dag.add_node(
        RiscOp::Expand {
            axis,
            size: axis_size.clone(),
        },
        vec![mean],
        x_ty.clone(),
    );
    let centered = lower_sub(dag, x, mean_expanded, x_ty);
    let squared = dag.add_node(RiscOp::Mul, vec![centered, centered], x_ty.clone());
    let var = lower_mean(dag, squared, axis, x_ty);
    let var_expanded = dag.add_node(
        RiscOp::Expand {
            axis,
            size: axis_size,
        },
        vec![var],
        x_ty.clone(),
    );
    let eps_const = dag.add_node(RiscOp::Const { value: eps }, vec![], x_ty.clone());
    let denom_sq = dag.add_node(RiscOp::Add, vec![var_expanded, eps_const], x_ty.clone());
    let denom = dag.add_node(RiscOp::Sqrt, vec![denom_sq], x_ty.clone());
    let normed = lower_div(dag, centered, denom, x_ty);

    let gamma_node = expand_to_match(dag, gamma, gamma_ty.clone(), &x_ty.dims);
    let beta_node = expand_to_match(dag, beta, beta_ty.clone(), &x_ty.dims);
    let scaled = dag.add_node(RiscOp::Mul, vec![normed, gamma_node], x_ty.clone());
    dag.add_node(RiscOp::Add, vec![scaled, beta_node], x_ty.clone())
}

/// conv2d(input, kernel, stride, padding) via a coarse im2col-style decomposition.
#[allow(clippy::too_many_arguments)]
pub fn lower_conv2d(
    dag: &mut Dag,
    input: NodeId,
    kernel: NodeId,
    input_ty: &TensorType,
    kernel_ty: &TensorType,
    output_ty: &TensorType,
    stride: usize,
    padding: usize,
) -> NodeId {
    let batch = require_dim(input_ty.dims.first(), "conv2d batch axis");
    let in_c = require_dim(input_ty.dims.get(1), "conv2d input channel axis");
    let h_in = require_dim(input_ty.dims.get(2), "conv2d input height axis");
    let w_in = require_dim(input_ty.dims.get(3), "conv2d input width axis");
    let out_c = require_dim(
        output_ty.dims.get(1).or_else(|| kernel_ty.dims.first()),
        "conv2d output channel axis",
    );
    let kh = require_dim(kernel_ty.dims.get(2), "conv2d kernel height axis");
    let kw = require_dim(kernel_ty.dims.get(3), "conv2d kernel width axis");
    let kh_size = require_dim_extent(&kh, "conv2d kernel height axis");
    let kw_size = require_dim_extent(&kw, "conv2d kernel width axis");

    let batch_size = require_dim_extent(&batch, "conv2d batch axis");
    let in_c_size = require_dim_extent(&in_c, "conv2d input channel axis");
    let h_in_size = require_dim_extent(&h_in, "conv2d input height axis");
    let w_in_size = require_dim_extent(&w_in, "conv2d input width axis");
    let stride = stride.max(1);
    let padded_h = h_in_size + (2 * padding);
    let padded_w = w_in_size + (2 * padding);
    if padded_h < kh_size || padded_w < kw_size {
        panic!(
            "Phase 0e conv2d kernel dims ({kh_size}, {kw_size}) exceed padded input dims ({padded_h}, {padded_w})"
        );
    }
    let strided_h = ((padded_h - kh_size) / stride) + 1;
    let strided_w = ((padded_w - kw_size) / stride) + 1;

    // Prefer the inferred `output_ty` spatial extents when concrete, but
    // fall back to the `(strided_h, strided_w)` arithmetic derived from
    // the input/kernel/stride/padding above. The fallback exists because
    // the type-annotation writeback is bottom-up: the inner `conv2d`
    // app's result-type metadata can remain as fresh non-concrete d-vars
    // even when the enclosing def's return-type ascription already pins
    // the spatial dims. See Phase 3j-pre Batch 3b notes in
    // `spec/design/chelis_phase3_plan.md`. The relaxation is strictly
    // concrete: we rebind `h_out`/`w_out` as `DimInfo::Lit` values.
    let raw_h_out = require_dim(output_ty.dims.get(2), "conv2d output height axis");
    let raw_w_out = require_dim(output_ty.dims.get(3), "conv2d output width axis");
    let h_out_size = match &raw_h_out {
        DimInfo::Lit(n) => *n,
        DimInfo::Named(_, Some(n)) => *n,
        DimInfo::Named(_, None) => strided_h,
    };
    let w_out_size = match &raw_w_out {
        DimInfo::Lit(n) => *n,
        DimInfo::Named(_, Some(n)) => *n,
        DimInfo::Named(_, None) => strided_w,
    };
    let h_out = DimInfo::Lit(h_out_size);
    let w_out = DimInfo::Lit(w_out_size);
    if h_out_size != strided_h || w_out_size != strided_w {
        panic!(
            "Phase 0e conv2d output shape mismatch: expected spatial dims ({strided_h}, {strided_w}), got ({h_out_size}, {w_out_size})"
        );
    }

    let padded_ty = TensorType {
        dims: vec![
            batch.clone(),
            in_c.clone(),
            DimInfo::Lit(padded_h),
            DimInfo::Lit(padded_w),
        ],
        precision: input_ty.precision,
    };
    let padded = dag.add_node(
        RiscOp::Pad {
            padding: vec![(0, 0), (0, 0), (padding, padding), (padding, padding)],
            fill: 0.0,
        },
        vec![input],
        padded_ty.clone(),
    );

    let sample_h = h_out_size.saturating_sub(1) * stride + 1;
    let sample_w = w_out_size.saturating_sub(1) * stride + 1;

    let mut acc: Option<NodeId> = None;
    for kh_idx in 0..kh_size {
        for kw_idx in 0..kw_size {
            let sampled = lower_conv2d_sample(
                dag,
                padded,
                &batch,
                &in_c,
                kh_idx,
                kw_idx,
                sample_h,
                sample_w,
                stride,
                input_ty.precision,
            );
            let sampled_ty = TensorType {
                dims: vec![batch.clone(), in_c.clone(), h_out.clone(), w_out.clone()],
                precision: input_ty.precision,
            };

            let kernel_slice = lower_conv2d_kernel_slice(
                dag,
                kernel,
                &out_c,
                &in_c,
                kh_idx,
                kw_idx,
                kernel_ty.precision,
            );
            let kernel_slice_ty = TensorType {
                dims: vec![
                    out_c.clone(),
                    in_c.clone(),
                    DimInfo::Lit(1),
                    DimInfo::Lit(1),
                ],
                precision: kernel_ty.precision,
            };

            let term = lower_conv2d_pointwise(
                dag,
                sampled,
                kernel_slice,
                &sampled_ty,
                &kernel_slice_ty,
                output_ty,
                batch_size,
                in_c_size,
                h_out.clone(),
                w_out.clone(),
                out_c.clone(),
            );

            acc = Some(match acc {
                Some(prev) => dag.add_node(RiscOp::Add, vec![prev, term], output_ty.clone()),
                None => term,
            });
        }
    }

    acc.expect("conv2d must emit at least one kernel contribution")
}

#[allow(clippy::too_many_arguments)]
fn lower_conv2d_sample(
    dag: &mut Dag,
    padded: NodeId,
    batch: &DimInfo,
    in_c: &DimInfo,
    kh_idx: usize,
    kw_idx: usize,
    sample_h: usize,
    sample_w: usize,
    stride: usize,
    precision: Prim,
) -> NodeId {
    let sampled_window_ty = TensorType {
        dims: vec![
            batch.clone(),
            in_c.clone(),
            DimInfo::Lit(sample_h),
            DimInfo::Lit(sample_w),
        ],
        precision,
    };
    let sampled_window = dag.add_node(
        RiscOp::Shrink {
            bounds: vec![
                (0, require_dim_extent(batch, "conv2d batch axis")),
                (0, require_dim_extent(in_c, "conv2d input channel axis")),
                (kh_idx, kh_idx + sample_h),
                (kw_idx, kw_idx + sample_w),
            ],
        },
        vec![padded],
        sampled_window_ty.clone(),
    );

    let sampled_ty = TensorType {
        dims: vec![
            batch.clone(),
            in_c.clone(),
            DimInfo::Lit(sample_h.div_ceil(stride)),
            DimInfo::Lit(sample_w.div_ceil(stride)),
        ],
        precision,
    };
    dag.add_node(
        RiscOp::Stride {
            strides: vec![1, 1, stride, stride],
        },
        vec![sampled_window],
        sampled_ty,
    )
}

fn lower_conv2d_kernel_slice(
    dag: &mut Dag,
    kernel: NodeId,
    out_c: &DimInfo,
    in_c: &DimInfo,
    kh_idx: usize,
    kw_idx: usize,
    precision: Prim,
) -> NodeId {
    dag.add_node(
        RiscOp::Shrink {
            bounds: vec![
                (0, require_dim_extent(out_c, "conv2d output channel axis")),
                (0, require_dim_extent(in_c, "conv2d input channel axis")),
                (kh_idx, kh_idx + 1),
                (kw_idx, kw_idx + 1),
            ],
        },
        vec![kernel],
        TensorType {
            dims: vec![
                out_c.clone(),
                in_c.clone(),
                DimInfo::Lit(1),
                DimInfo::Lit(1),
            ],
            precision,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn lower_conv2d_pointwise(
    dag: &mut Dag,
    input: NodeId,
    kernel: NodeId,
    input_ty: &TensorType,
    kernel_ty: &TensorType,
    output_ty: &TensorType,
    batch_size: usize,
    in_c_size: usize,
    h_out: DimInfo,
    w_out: DimInfo,
    out_c: DimInfo,
) -> NodeId {
    let permuted_ty = TensorType {
        dims: vec![
            input_ty.dims[1].clone(),
            input_ty.dims[0].clone(),
            input_ty.dims[2].clone(),
            input_ty.dims[3].clone(),
        ],
        precision: input_ty.precision,
    };
    let permuted = dag.add_node(
        RiscOp::Permute {
            axes: vec![1, 0, 2, 3],
        },
        vec![input],
        permuted_ty.clone(),
    );

    let h_out_size = require_dim_extent(&h_out, "conv2d output height axis");
    let w_out_size = require_dim_extent(&w_out, "conv2d output width axis");
    let col_dim = DimInfo::Lit(batch_size * h_out_size * w_out_size);
    let cols_ty = TensorType {
        dims: vec![input_ty.dims[1].clone(), col_dim.clone()],
        precision: input_ty.precision,
    };
    let cols = dag.add_node(
        RiscOp::Reshape {
            new_shape: cols_ty.dims.clone(),
        },
        vec![permuted],
        cols_ty.clone(),
    );

    let kernel_flat_ty = TensorType {
        dims: vec![out_c.clone(), DimInfo::Lit(in_c_size)],
        precision: kernel_ty.precision,
    };
    let kernel_flat = dag.add_node(
        RiscOp::Reshape {
            new_shape: kernel_flat_ty.dims.clone(),
        },
        vec![kernel],
        kernel_flat_ty.clone(),
    );

    let product = lower_matmul(dag, kernel_flat, cols, &kernel_flat_ty, &cols_ty);
    let product_4d_ty = TensorType {
        dims: vec![
            out_c.clone(),
            input_ty.dims[0].clone(),
            h_out.clone(),
            w_out.clone(),
        ],
        precision: output_ty.precision,
    };
    let product_4d = dag.add_node(
        RiscOp::Reshape {
            new_shape: product_4d_ty.dims.clone(),
        },
        vec![product],
        product_4d_ty,
    );
    dag.add_node(
        RiscOp::Permute {
            axes: vec![1, 0, 2, 3],
        },
        vec![product_4d],
        output_ty.clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::RiscOp;
    use crate::verify;

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    #[test]
    fn sub_produces_add_neg() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let result = lower_sub(&mut dag, a, b, &scalar_f32());
        assert!(verify::verify(&dag).is_empty());

        // Should have: Const(5), Const(3), Neg, Add
        assert_eq!(dag.len(), 4);
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Add);
        // The Neg node is one of the inputs to Add.
        let neg_id = result_node.inputs[1];
        assert_eq!(dag.get(neg_id).unwrap().op, RiscOp::Neg);
    }

    #[test]
    fn relu_produces_max_elem_const_zero() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: -1.0 }, vec![], scalar_f32());
        let result = lower_relu(&mut dag, x, &scalar_f32());
        assert!(verify::verify(&dag).is_empty());

        // Const(-1), Const(0), MaxElem
        assert_eq!(dag.len(), 3);
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::MaxElem);
        let zero_id = result_node.inputs[1];
        assert_eq!(dag.get(zero_id).unwrap().op, RiscOp::Const { value: 0.0 });
    }

    #[test]
    fn sigmoid_produces_correct_chain() {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let result = lower_sigmoid(&mut dag, x, &scalar_f32());
        assert!(verify::verify(&dag).is_empty());

        // x, neg(x), exp(neg(x)), const(1), add, log, neg, exp
        assert_eq!(dag.len(), 8);
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Exp);
    }

    #[test]
    fn div_produces_mul_recip() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 6.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let result = lower_div(&mut dag, a, b, &scalar_f32());
        assert!(verify::verify(&dag).is_empty());

        // a, b, log(b), neg(log(b)), exp(neg(log(b))), mul(a, recip(b))
        assert_eq!(dag.len(), 6);
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Mul);
    }

    // --- H1: Tier 2 comparison decompositions ---

    fn scalar_bool() -> TensorType {
        TensorType {
            dims: vec![],
            precision: Prim::Bool,
        }
    }

    #[test]
    fn gt_swaps_args_to_cmplt() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let result = lower_gt(&mut dag, a, b, &scalar_f32());

        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        // b, a order (swapped).
        assert_eq!(node.inputs, vec![b, a]);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn gte_produces_not_cmplt() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let result = lower_gte(&mut dag, a, b, &scalar_f32());

        // a, b, CmpLt(a,b), Const(1), CmpLt(lt, 1)
        assert_eq!(dag.len(), 5);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn lte_produces_not_cmplt_ba() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let result = lower_lte(&mut dag, a, b, &scalar_f32());

        assert_eq!(dag.len(), 5);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn eq_produces_not_or_cmplt() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let result = lower_eq(&mut dag, a, b, &scalar_f32());

        // a, b, CmpLt(a,b), CmpLt(b,a), MaxElem, Const(1), CmpLt(or, 1)
        assert_eq!(dag.len(), 7);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn neq_produces_or_cmplt() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let result = lower_neq(&mut dag, a, b, &scalar_f32());

        // a, b, CmpLt(a,b), CmpLt(b,a), MaxElem
        assert_eq!(dag.len(), 5);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::MaxElem);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn min_elem_produces_neg_max_neg() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let result = lower_min_elem(&mut dag, a, b, &scalar_f32());
        assert!(verify::verify(&dag).is_empty());

        // a, b, neg(a), neg(b), max(neg_a, neg_b), neg(max)
        assert_eq!(dag.len(), 6);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::Neg);
    }

    // --- H2: Boolean operators ---

    #[test]
    fn and_produces_mul() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_bool());
        let b = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_bool());
        let result = lower_and(&mut dag, a, b, &scalar_bool());

        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::Mul);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn or_produces_max_elem() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_bool());
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_bool());
        let result = lower_or(&mut dag, a, b, &scalar_bool());

        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::MaxElem);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn not_produces_cmplt_with_one() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_bool());
        let result = lower_not(&mut dag, a, &scalar_bool());

        // a, Const(1), CmpLt(a, 1)
        assert_eq!(dag.len(), 3);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    // --- Higher-level decompositions ---

    fn matrix_2x3() -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: Prim::F32,
        }
    }

    fn matrix_3x4() -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        }
    }

    fn vec_5() -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(5)],
            precision: Prim::F32,
        }
    }

    #[test]
    fn matmul_produces_expand_mul_sum() {
        let mut dag = Dag::new();
        let a_ty = matrix_2x3();
        let b_ty = matrix_3x4();
        let a = dag.add_node(RiscOp::Load { name: "A".into() }, vec![], a_ty.clone());
        let b = dag.add_node(RiscOp::Load { name: "B".into() }, vec![], b_ty.clone());
        let result = lower_matmul(&mut dag, a, b, &a_ty, &b_ty);

        // A, B, Expand(A), Expand(B), Mul, Sum
        assert_eq!(dag.len(), 6);

        // Check Expand nodes exist
        let ops: Vec<_> = dag.nodes().iter().map(|n| &n.op).collect();
        let expand_count = ops
            .iter()
            .filter(|op| matches!(op, RiscOp::Expand { .. }))
            .count();
        assert_eq!(expand_count, 2, "expected 2 Expand nodes");
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Mul)),
            "expected a Mul node"
        );
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Sum { axis: 1 })),
            "expected a Sum{{axis:1}} node"
        );

        // Result type should be [2, 4]
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.output_type.dims.len(), 2);
        assert_eq!(result_node.output_type.dims[0], DimInfo::Lit(2));
        assert_eq!(result_node.output_type.dims[1], DimInfo::Lit(4));
    }

    #[test]
    fn softmax_produces_maxreduce_sub_exp_sum_div() {
        let mut dag = Dag::new();
        let ty = vec_5();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone());
        let result = lower_softmax(&mut dag, x, 0, &ty);

        // Check the chain of ops produced.
        let ops: Vec<_> = dag.nodes().iter().map(|n| &n.op).collect();
        assert!(
            ops.iter()
                .any(|op| matches!(op, RiscOp::MaxReduce { axis: 0 })),
            "expected MaxReduce"
        );
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Exp)),
            "expected Exp"
        );
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Sum { axis: 0 })),
            "expected Sum"
        );
        // Sub produces Add+Neg, Div produces Log+Neg+Exp+Mul
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Neg)),
            "expected Neg (from sub)"
        );

        // The final result should be a Mul (from div decomposition)
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Mul);
    }

    #[test]
    fn mean_produces_sum_div() {
        let mut dag = Dag::new();
        let ty = vec_5();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone());
        let result = lower_mean(&mut dag, x, 0, &ty);

        let ops: Vec<_> = dag.nodes().iter().map(|n| &n.op).collect();
        // Should have Sum
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Sum { axis: 0 })),
            "expected Sum"
        );
        // Should have Const(5.0) for dim size
        assert!(
            ops.iter()
                .any(|op| matches!(op, RiscOp::Const { value } if *value == 5.0)),
            "expected Const(5.0) for dimension size"
        );
        // Result is Mul (from div decomposition: mul(sum, recip(5)))
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Mul);
    }

    #[test]
    fn layer_norm_produces_mean_variance_and_affine_ops() {
        let mut dag = Dag::new();
        let x_ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let scale_ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], x_ty.clone());
        let gamma = dag.add_node(
            RiscOp::Load {
                name: "gamma".into(),
            },
            vec![],
            scale_ty.clone(),
        );
        let beta = dag.add_node(
            RiscOp::Load {
                name: "beta".into(),
            },
            vec![],
            scale_ty.clone(),
        );
        let result = lower_layer_norm(&mut dag, x, gamma, beta, &x_ty, &scale_ty, &scale_ty, 1e-5);

        let ops: Vec<_> = dag.nodes().iter().map(|n| &n.op).collect();
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Sum { axis: 1 })),
            "expected a reduction over the hidden axis"
        );
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Sqrt)),
            "expected sqrt in the denominator"
        );
        let expand_count = ops
            .iter()
            .filter(|op| matches!(op, RiscOp::Expand { .. }))
            .count();
        assert!(
            expand_count >= 4,
            "expected expansion of mean/var/gamma/beta"
        );
        assert_eq!(dag.get(result).unwrap().output_type, x_ty);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn layer_norm_expands_rank3_suffix_correctly() {
        let mut dag = Dag::new();
        let x_ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let scale_ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], x_ty.clone());
        let gamma = dag.add_node(
            RiscOp::Load {
                name: "gamma".into(),
            },
            vec![],
            scale_ty.clone(),
        );
        let beta = dag.add_node(
            RiscOp::Load {
                name: "beta".into(),
            },
            vec![],
            scale_ty.clone(),
        );
        let _ = lower_layer_norm(&mut dag, x, gamma, beta, &x_ty, &scale_ty, &scale_ty, 1e-5);
        assert!(
            dag.nodes().iter().any(|node| {
                matches!(node.op, RiscOp::Expand { axis: 0, .. })
                    && node.output_type.dims
                        == vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)]
            }),
            "expected gamma/beta expansion to align with trailing hidden dim"
        );
    }

    #[test]
    fn conv2d_produces_im2col_style_sequence() {
        let mut dag = Dag::new();
        let input_ty = TensorType {
            dims: vec![
                DimInfo::Lit(1),
                DimInfo::Lit(3),
                DimInfo::Lit(8),
                DimInfo::Lit(8),
            ],
            precision: Prim::F32,
        };
        let kernel_ty = TensorType {
            dims: vec![
                DimInfo::Lit(16),
                DimInfo::Lit(3),
                DimInfo::Lit(2),
                DimInfo::Lit(2),
            ],
            precision: Prim::F32,
        };
        let output_ty = TensorType {
            dims: vec![
                DimInfo::Lit(1),
                DimInfo::Lit(16),
                DimInfo::Lit(7),
                DimInfo::Lit(7),
            ],
            precision: Prim::F32,
        };
        let input = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty.clone());
        let kernel = dag.add_node(RiscOp::Load { name: "w".into() }, vec![], kernel_ty.clone());
        let result = lower_conv2d(
            &mut dag, input, kernel, &input_ty, &kernel_ty, &output_ty, 1, 0,
        );

        let ops: Vec<_> = dag.nodes().iter().map(|n| &n.op).collect();
        assert!(ops.iter().any(|op| matches!(op, RiscOp::Pad { .. })));
        assert!(ops.iter().any(|op| matches!(op, RiscOp::Stride { .. })));
        assert!(ops.iter().any(|op| matches!(op, RiscOp::Shrink { .. })));
        assert!(ops.iter().any(|op| matches!(op, RiscOp::Permute { .. })));
        assert!(
            ops.iter()
                .filter(|op| matches!(op, RiscOp::Reshape { .. }))
                .count()
                >= 3,
            "expected im2col reshapes plus output reshape"
        );
        assert!(ops.iter().any(|op| matches!(op, RiscOp::Mul)));
        assert!(ops.iter().any(|op| matches!(op, RiscOp::Sum { .. })));
        assert_eq!(dag.get(result).unwrap().output_type, output_ty);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn softmax_accepts_symbolic_axis_extent() {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Named("batch".into(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone());
        let out = lower_softmax(&mut dag, x, 0, &ty);
        let expand_sizes: Vec<_> = dag
            .nodes()
            .iter()
            .filter_map(|node| match &node.op {
                RiscOp::Expand { size, .. } => Some(size.clone()),
                _ => None,
            })
            .collect();

        assert!(
            expand_sizes
                .iter()
                .all(|size| matches!(size, DimExpr::Sym(name) if name == "batch")),
            "softmax should preserve a symbolic axis extent through expand nodes"
        );
        assert_eq!(dag.get(out).unwrap().output_type, ty);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    #[should_panic(expected = "layer_norm requires a concrete extent")]
    fn layer_norm_rejects_symbolic_normalized_axis_extent() {
        let mut dag = Dag::new();
        let x_ty = TensorType {
            dims: vec![DimInfo::Named("hidden".into(), None)],
            precision: Prim::F32,
        };
        let scale_ty = TensorType {
            dims: vec![DimInfo::Named("hidden".into(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], x_ty.clone());
        let gamma = dag.add_node(
            RiscOp::Load {
                name: "gamma".into(),
            },
            vec![],
            scale_ty.clone(),
        );
        let beta = dag.add_node(
            RiscOp::Load {
                name: "beta".into(),
            },
            vec![],
            scale_ty.clone(),
        );
        let _ = lower_layer_norm(&mut dag, x, gamma, beta, &x_ty, &scale_ty, &scale_ty, 1e-5);
    }

    #[test]
    #[should_panic(expected = "conv2d output height axis requires a statically known axis")]
    fn conv2d_rejects_missing_output_shape() {
        let mut dag = Dag::new();
        let input_ty = TensorType {
            dims: vec![
                DimInfo::Lit(1),
                DimInfo::Lit(3),
                DimInfo::Lit(8),
                DimInfo::Lit(8),
            ],
            precision: Prim::F32,
        };
        let kernel_ty = TensorType {
            dims: vec![
                DimInfo::Lit(16),
                DimInfo::Lit(3),
                DimInfo::Lit(1),
                DimInfo::Lit(1),
            ],
            precision: Prim::F32,
        };
        let output_ty = TensorType {
            dims: vec![DimInfo::Lit(1), DimInfo::Lit(16)],
            precision: Prim::F32,
        };
        let input = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty.clone());
        let kernel = dag.add_node(RiscOp::Load { name: "w".into() }, vec![], kernel_ty.clone());
        let _ = lower_conv2d(
            &mut dag, input, kernel, &input_ty, &kernel_ty, &output_ty, 1, 1,
        );
    }
}
