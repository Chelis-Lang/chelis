//! Tier 2 decomposition helpers and direct identity lowerers.
//!
//! Most functions in this module decompose Tier 2 derived operations into
//! Tier 1 RISC DAG nodes. `lower_sub` and `lower_min_elem` emit direct Tier-1 identities
//! while retaining this module's shared span-propagation path.
//!
//! For decomposing helpers, span propagation follows
//! `spec/design/chelis_span_survival.md` §2.3's Tier 2 row: every synthesized
//! sub-node inherits the decomposed parent's
//! `span_id`. If the parent had no span, sub-nodes carry the canonical
//! `__synthesized_tier2__` marker (defined in
//! `spec/03-deep-syntax.md` §1.1.1).
//!
//! Each public lowerer takes `parent_span: Option<&str>`:
//!   * `Some(s)` — the operation's source span; emitted nodes inherit `s`.
//!   * `None` — no source region; emitted nodes carry `__synthesized_tier2__`.
//!
//! The internal `add_synth` helper applies the rule once per node so we
//! don't duplicate it across the ~109 `add_node` callsites.

use chelis_types::types::Prim;

use crate::dag::{Dag, DimExpr, DimInfo, NodeId, RiscOp, RtAxis, RtDim, TensorType};

/// Canonical synthesized marker for Tier 2 decomposition sub-nodes when
/// the parent op had no source span. Locked by spec/03-deep-syntax.md
/// §1.1.1 — double-underscore wrap, lowercase pass name.
pub const TIER2_SYNTH_MARKER: &str = "__synthesized_tier2__";

/// Internal `add_node` wrapper that threads the parent span (or applies
/// the `__synthesized_tier2__` marker) onto every Tier 2 sub-node. All
/// Tier 2 helpers route their `add_node` calls through this so the
/// rule is applied uniformly.
fn add_synth(
    dag: &mut Dag,
    op: RiscOp,
    inputs: Vec<NodeId>,
    output_type: TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    let span_id = match parent_span {
        Some(s) => Some(s.to_owned()),
        None => Some(TIER2_SYNTH_MARKER.to_owned()),
    };
    dag.add_node(op, inputs, output_type, span_id)
}

/// Direct checked `sub(a, b)` identity ([05-OP-41]).
pub fn lower_sub(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    add_synth(dag, RiscOp::Sub, vec![a, b], ty.clone(), parent_span)
}

/// Preserve the [05-OP-43] `relu` identity through AD. Its forward value is
/// `max_elem(x, const(0))`, but MaxElem's first-operand tie adjoint is not the
/// ReLU convention at zero.
pub fn lower_relu(dag: &mut Dag, x: NodeId, ty: &TensorType, parent_span: Option<&str>) -> NodeId {
    add_synth(dag, RiscOp::Relu, vec![x], ty.clone(), parent_span)
}

/// `sigmoid(x)` = `1 / (1 + exp(-x))`
///
/// Lowered as `recip(add(const(1), exp(neg(x))))`. The reciprocal
/// step is a single `RiscOp::Recip` so the lowering produces the
/// IEEE-correct value (`1 / 0 = +inf`) rather than the
/// NaN-from-log that an `exp(neg(log(_)))` decomposition would
/// produce on non-positive inputs.
pub fn lower_sigmoid(
    dag: &mut Dag,
    x: NodeId,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    let neg_x = add_synth(dag, RiscOp::Neg, vec![x], ty.clone(), parent_span);
    let exp_neg = add_synth(dag, RiscOp::Exp, vec![neg_x], ty.clone(), parent_span);
    let one = add_synth(
        dag,
        RiscOp::synth_const(ty.precision, 1.0),
        vec![],
        ty.clone(),
        parent_span,
    );
    let sum = add_synth(
        dag,
        RiscOp::Add,
        vec![one, exp_neg],
        ty.clone(),
        parent_span,
    );
    add_synth(dag, RiscOp::Recip, vec![sum], ty.clone(), parent_span)
}

/// `tanh(x)` = `2 * sigmoid(2*x) - 1`
///
/// Decomposed via existing RISC primitives (Exp/Add/Mul/Neg/Log) so we
/// avoid escalating the RISC vocabulary. The C backend numerics match
/// the host-runtime helper `chelis_host_tanh_f32` to f32 ulp tolerance
/// (both routes ultimately go through `expf`).
pub fn lower_tanh(dag: &mut Dag, x: NodeId, ty: &TensorType, parent_span: Option<&str>) -> NodeId {
    let two = add_synth(
        dag,
        RiscOp::synth_const(ty.precision, 2.0),
        vec![],
        ty.clone(),
        parent_span,
    );
    let two_x = add_synth(dag, RiscOp::Mul, vec![two, x], ty.clone(), parent_span);
    let sig_2x = lower_sigmoid(dag, two_x, ty, parent_span);
    let two_again = add_synth(
        dag,
        RiscOp::synth_const(ty.precision, 2.0),
        vec![],
        ty.clone(),
        parent_span,
    );
    let two_sig = add_synth(
        dag,
        RiscOp::Mul,
        vec![two_again, sig_2x],
        ty.clone(),
        parent_span,
    );
    let neg_one = add_synth(
        dag,
        RiscOp::synth_const(ty.precision, -1.0),
        vec![],
        ty.clone(),
        parent_span,
    );
    add_synth(
        dag,
        RiscOp::Add,
        vec![two_sig, neg_one],
        ty.clone(),
        parent_span,
    )
}

/// `silu(x)` = `x * sigmoid(x)` (a.k.a. swish).
pub fn lower_silu(dag: &mut Dag, x: NodeId, ty: &TensorType, parent_span: Option<&str>) -> NodeId {
    let sig_x = lower_sigmoid(dag, x, ty, parent_span);
    add_synth(dag, RiscOp::Mul, vec![x, sig_x], ty.clone(), parent_span)
}

/// `gelu(x)` via the tanh approximation:
///
///   gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
///
/// Matches `School.Nn.Gelu.gelu_scalar` and the host-runtime helper
/// `activation_gelu_f32`. If/when an `Erf` RISC op lands, the
/// erf-exact form can replace this — both lanes must move together.
pub fn lower_gelu(dag: &mut Dag, x: NodeId, ty: &TensorType, parent_span: Option<&str>) -> NodeId {
    // c = sqrt(2/pi)
    let c = add_synth(
        dag,
        RiscOp::synth_const(ty.precision, 0.7978845608028654),
        vec![],
        ty.clone(),
        parent_span,
    );
    let k = add_synth(
        dag,
        RiscOp::synth_const(ty.precision, 0.044715),
        vec![],
        ty.clone(),
        parent_span,
    );
    // x^3 = x * x * x
    let x_sq = add_synth(dag, RiscOp::Mul, vec![x, x], ty.clone(), parent_span);
    let x_cu = add_synth(dag, RiscOp::Mul, vec![x_sq, x], ty.clone(), parent_span);
    // k * x^3
    let k_x_cu = add_synth(dag, RiscOp::Mul, vec![k, x_cu], ty.clone(), parent_span);
    // x + k * x^3
    let sum_inner = add_synth(dag, RiscOp::Add, vec![x, k_x_cu], ty.clone(), parent_span);
    // c * (x + k * x^3)
    let inner = add_synth(
        dag,
        RiscOp::Mul,
        vec![c, sum_inner],
        ty.clone(),
        parent_span,
    );
    let tanh_inner = lower_tanh(dag, inner, ty, parent_span);
    // 1 + tanh(inner)
    let one = add_synth(
        dag,
        RiscOp::synth_const(ty.precision, 1.0),
        vec![],
        ty.clone(),
        parent_span,
    );
    let one_plus_tanh = add_synth(
        dag,
        RiscOp::Add,
        vec![one, tanh_inner],
        ty.clone(),
        parent_span,
    );
    // x * (1 + tanh(inner))
    let x_mul = add_synth(
        dag,
        RiscOp::Mul,
        vec![x, one_plus_tanh],
        ty.clone(),
        parent_span,
    );
    // 0.5 * x * (1 + tanh(inner))
    let half = add_synth(
        dag,
        RiscOp::synth_const(ty.precision, 0.5),
        vec![],
        ty.clone(),
        parent_span,
    );
    add_synth(dag, RiscOp::Mul, vec![half, x_mul], ty.clone(), parent_span)
}

/// `div(a, b)` — IEEE-754 elementwise division.
///
/// Lowers directly to `RiscOp::Div`. An algebraic `mul(a,
/// exp(neg(log(b))))` rewrite is only valid for `b > 0`; for
/// `b ≤ 0` `log(b)` is undefined and the result is NaN. The
/// primitive delegates to native IEEE `/` on every supported
/// target.
pub fn lower_div(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    add_synth(dag, RiscOp::Div, vec![a, b], ty.clone(), parent_span)
}

/// `floor_div(a, b)` — floor division (round quotient toward −∞).
///
/// Lowers directly to [`RiscOp::FloorDiv`] (chelis#178). Integer
/// operands round toward −∞; float operands compute `floor(a / b)`.
pub fn lower_floor_div(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    add_synth(dag, RiscOp::FloorDiv, vec![a, b], ty.clone(), parent_span)
}

/// `trunc_div(a, b)` — truncating (round-toward-zero) integer division.
///
/// Lowers directly to [`RiscOp::TruncDiv`] (chelis#178). Integer
/// operands only; the C/Rust integer `/` quotient.
pub fn lower_trunc_div(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    add_synth(dag, RiscOp::TruncDiv, vec![a, b], ty.clone(), parent_span)
}

/// H1: `gt(a, b)` = `cmplt(b, a)` (swap args)
pub fn lower_gt(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    add_synth(dag, RiscOp::CmpLt, vec![b, a], bool_ty, parent_span)
}

/// H1: `gte(a, b)` = `neg(cmplt(a, b))` — not (a < b)
/// Per spec §3.2: gte uses neg on bool (0/1 convention).
pub fn lower_gte(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    let lt = add_synth(dag, RiscOp::CmpLt, vec![a, b], bool_ty.clone(), parent_span);
    // not(lt): cmplt(lt, const(1)) — if lt==0 then 0<1=true, if lt==1 then 1<1=false
    let one = add_synth(
        dag,
        RiscOp::synth_const(bool_ty.precision, 1.0),
        vec![],
        bool_ty.clone(),
        parent_span,
    );
    add_synth(dag, RiscOp::CmpLt, vec![lt, one], bool_ty, parent_span)
}

/// H1: `lte(a, b)` = `neg(cmplt(b, a))` — not (b < a)
pub fn lower_lte(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    let lt = add_synth(dag, RiscOp::CmpLt, vec![b, a], bool_ty.clone(), parent_span);
    let one = add_synth(
        dag,
        RiscOp::synth_const(bool_ty.precision, 1.0),
        vec![],
        bool_ty.clone(),
        parent_span,
    );
    add_synth(dag, RiscOp::CmpLt, vec![lt, one], bool_ty, parent_span)
}

/// H1: `eq(a, b)` = not(or(cmplt(a,b), cmplt(b,a)))
/// `or` on bools = `max_elem`, `not` = `cmplt(x, const(1))`
pub fn lower_eq(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    let lt_ab = add_synth(dag, RiscOp::CmpLt, vec![a, b], bool_ty.clone(), parent_span);
    let lt_ba = add_synth(dag, RiscOp::CmpLt, vec![b, a], bool_ty.clone(), parent_span);
    let or = add_synth(
        dag,
        RiscOp::MaxElem,
        vec![lt_ab, lt_ba],
        bool_ty.clone(),
        parent_span,
    );
    let one = add_synth(
        dag,
        RiscOp::synth_const(bool_ty.precision, 1.0),
        vec![],
        bool_ty.clone(),
        parent_span,
    );
    add_synth(dag, RiscOp::CmpLt, vec![or, one], bool_ty, parent_span)
}

/// H1: `neq(a, b)` = `or(cmplt(a,b), cmplt(b,a))`
pub fn lower_neq(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    let lt_ab = add_synth(dag, RiscOp::CmpLt, vec![a, b], bool_ty.clone(), parent_span);
    let lt_ba = add_synth(dag, RiscOp::CmpLt, vec![b, a], bool_ty.clone(), parent_span);
    add_synth(
        dag,
        RiscOp::MaxElem,
        vec![lt_ab, lt_ba],
        bool_ty,
        parent_span,
    )
}

/// Direct stored-bit `min_elem(a, b)` selection identity ([05-OP-40]).
pub fn lower_min_elem(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    add_synth(dag, RiscOp::MinElem, vec![a, b], ty.clone(), parent_span)
}

/// H2: `and(a, b)` on bools = `mul(a, b)`
pub fn lower_and(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    add_synth(dag, RiscOp::Mul, vec![a, b], bool_ty, parent_span)
}

/// H2: `or(a, b)` on bools = `max_elem(a, b)`
pub fn lower_or(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    add_synth(dag, RiscOp::MaxElem, vec![a, b], bool_ty, parent_span)
}

/// H2: `not(a)` on bools = `cmplt(a, const(1))` — flips 0->1, 1->0
pub fn lower_not(dag: &mut Dag, a: NodeId, ty: &TensorType, parent_span: Option<&str>) -> NodeId {
    let bool_ty = TensorType {
        dims: ty.dims.clone(),
        precision: Prim::Bool,
    };
    let one = add_synth(
        dag,
        RiscOp::synth_const(bool_ty.precision, 1.0),
        vec![],
        bool_ty.clone(),
        parent_span,
    );
    add_synth(dag, RiscOp::CmpLt, vec![a, one], bool_ty, parent_span)
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

fn dim_known_size(dim: &DimInfo) -> Option<usize> {
    match dim {
        DimInfo::Lit(size) | DimInfo::Named(_, Some(size)) => Some(*size),
        DimInfo::Named(_, None) => None,
    }
}

fn rt_axis(axis: usize) -> RtAxis {
    RtAxis::Lit(i32::try_from(axis).expect("tensor rank fits the int32 axis carrier"))
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
    dim.cloned()
        .unwrap_or_else(|| panic!("{context} requires a statically known axis in IR lowering"))
}

fn require_dim_extent(dim: &DimInfo, context: &str) -> usize {
    // IR lowering assumes dimensions are concrete by this point.
    // Named dimensions are for type-checking consistency; symbolic/runtime-sized
    // lowering is a Phase 1 IR/backend extension.
    match dim {
        DimInfo::Lit(n) => *n,
        DimInfo::Named(_, Some(n)) => *n,
        DimInfo::Named(name, None) => {
            panic!("{context} requires a concrete extent in IR lowering, got `{name}`")
        }
    }
}

fn require_axis_size(ty: &TensorType, axis: usize, context: &str) -> usize {
    dim_size(ty, axis).unwrap_or_else(|| {
        panic!("{context} requires a concrete extent for axis {axis} in IR lowering")
    })
}

fn expand_to_match(
    dag: &mut Dag,
    mut node: NodeId,
    mut ty: TensorType,
    target: NodeId,
    target_dims: &[DimInfo],
    parent_span: Option<&str>,
) -> NodeId {
    while ty.dims.len() < target_dims.len() {
        let missing = target_dims.len() - ty.dims.len();
        let insert_at = 0;
        let source_dim = target_dims[missing - 1].clone();
        let size = match dim_known_size(&source_dim) {
            Some(size) => RtDim::Lit(size),
            None => RtDim::InputAxis {
                tensor: 1,
                axis: rt_axis(missing - 1),
            },
        };
        let mut next_dims = ty.dims.clone();
        next_dims.insert(insert_at, source_dim);
        let next_ty = TensorType {
            dims: next_dims,
            precision: ty.precision,
        };
        node = add_synth(
            dag,
            RiscOp::Expand {
                axis: insert_at,
                size: size.clone(),
            },
            if matches!(size, RtDim::Lit(_)) {
                vec![node]
            } else {
                vec![node, target]
            },
            next_ty.clone(),
            parent_span,
        );
        ty = next_ty;
    }
    node
}

// ---------------------------------------------------------------------------
// Tier 2 higher-level decompositions (spec §3.4, §4.1–4.2)
// ---------------------------------------------------------------------------

/// matmul(A: [..., i, j], B: [..., j, k]) -> [..., i, k]
///
/// Lowering (spec §4.1):
///   1. Broadcast leading axes.
///   2. Expand A from [..., i, j] to [..., i, j, k].
///   3. Expand B from [..., j, k] to [..., i, j, k].
///   4. Mul the expanded tensors -> [..., i, j, k].
///   5. Sum over the j dimension -> [..., i, k].
///
pub fn lower_matmul(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    a_ty: &TensorType,
    b_ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    assert!(
        a_ty.dims.len() >= 2 && b_ty.dims.len() >= 2,
        "lower_matmul expects rank >= 2 tensors"
    );
    let a_lead = &a_ty.dims[..a_ty.dims.len() - 2];
    let b_lead = &b_ty.dims[..b_ty.dims.len() - 2];
    let lead_dims = broadcast_leading_dims(a_lead, b_lead);
    let lead_len = lead_dims.len();
    let i_dim = require_dim(a_ty.dims.get(a_ty.dims.len() - 2), "matmul lhs row axis");
    let j_dim = require_dim(
        a_ty.dims
            .last()
            .or_else(|| b_ty.dims.get(b_ty.dims.len() - 2)),
        "matmul shared axis",
    );
    let k_dim = require_dim(b_ty.dims.last(), "matmul rhs col axis");
    let i_size = match dim_known_size(&i_dim) {
        Some(size) => RtDim::Lit(size),
        None => RtDim::InputAxis {
            tensor: 1,
            axis: rt_axis(a_ty.dims.len() - 2),
        },
    };
    let k_size = match dim_known_size(&k_dim) {
        Some(size) => RtDim::Lit(size),
        None => RtDim::InputAxis {
            tensor: 1,
            axis: rt_axis(b_ty.dims.len() - 1),
        },
    };

    // The intermediate expanded type is [..., i, j, k].
    let mut expanded_dims = lead_dims.clone();
    expanded_dims.extend([i_dim.clone(), j_dim.clone(), k_dim.clone()]);
    let expanded_ty = TensorType {
        dims: expanded_dims,
        precision: a_ty.precision,
    };

    // The result type is [..., i, k].
    let mut result_dims = lead_dims.clone();
    result_dims.extend([i_dim.clone(), k_dim.clone()]);
    let result_ty = TensorType {
        dims: result_dims,
        precision: a_ty.precision,
    };

    let a_aligned = align_matmul_operand(
        dag,
        a,
        a_ty,
        (b, b_ty),
        &lead_dims,
        &[i_dim.clone(), j_dim.clone()],
        parent_span,
    );
    let a_expanded = add_synth(
        dag,
        RiscOp::Expand {
            axis: lead_len + 2,
            size: k_size.clone(),
        },
        if matches!(k_size, RtDim::Lit(_)) {
            vec![a_aligned]
        } else {
            vec![a_aligned, b]
        },
        expanded_ty.clone(),
        parent_span,
    );

    let b_aligned = align_matmul_operand(
        dag,
        b,
        b_ty,
        (a, a_ty),
        &lead_dims,
        &[j_dim.clone(), k_dim.clone()],
        parent_span,
    );
    let b_with_i = add_synth(
        dag,
        RiscOp::Expand {
            axis: lead_len,
            size: i_size.clone(),
        },
        if matches!(i_size, RtDim::Lit(_)) {
            vec![b_aligned]
        } else {
            vec![b_aligned, a]
        },
        expanded_ty.clone(),
        parent_span,
    );

    let product = add_synth(
        dag,
        RiscOp::Mul,
        vec![a_expanded, b_with_i],
        expanded_ty,
        parent_span,
    );

    add_synth(
        dag,
        RiscOp::sum_default(lead_len + 1, a_ty.precision)
            .expect("lower_matmul operand precision should accept reduce_sum"),
        vec![product],
        result_ty,
        parent_span,
    )
}

fn broadcast_leading_dims(lhs: &[DimInfo], rhs: &[DimInfo]) -> Vec<DimInfo> {
    let len = lhs.len().max(rhs.len());
    let mut out = Vec::with_capacity(len);
    for offset in 0..len {
        let lhs_idx = lhs.len().checked_sub(len - offset);
        let rhs_idx = rhs.len().checked_sub(len - offset);
        let dim = match (lhs_idx.map(|idx| &lhs[idx]), rhs_idx.map(|idx| &rhs[idx])) {
            (Some(a), Some(b)) if is_one_dim(a) => b.clone(),
            (Some(a), Some(b)) if is_one_dim(b) => a.clone(),
            (Some(a), Some(_)) => a.clone(),
            (Some(a), None) => a.clone(),
            (None, Some(b)) => b.clone(),
            (None, None) => unreachable!(),
        };
        out.push(dim);
    }
    out
}

fn is_one_dim(dim: &DimInfo) -> bool {
    matches!(dim, DimInfo::Lit(1) | DimInfo::Named(_, Some(1)))
}

fn align_matmul_operand(
    dag: &mut Dag,
    node: NodeId,
    ty: &TensorType,
    counterpart: (NodeId, &TensorType),
    lead_dims: &[DimInfo],
    matrix_dims: &[DimInfo; 2],
    parent_span: Option<&str>,
) -> NodeId {
    let (counterpart, counterpart_ty) = counterpart;
    let source_lead = &ty.dims[..ty.dims.len() - 2];
    let mut current = node;
    let mut current_dims = ty.dims.clone();
    let missing = lead_dims.len().saturating_sub(source_lead.len());
    for (axis, lead_dim) in lead_dims.iter().take(missing).enumerate() {
        let counterpart_lead_len = counterpart_ty.dims.len() - 2;
        let counterpart_axis = axis
            .checked_sub(lead_dims.len().saturating_sub(counterpart_lead_len))
            .expect("a missing matmul lead axis must be supplied by the counterpart");
        let size = match dim_known_size(lead_dim) {
            Some(size) => RtDim::Lit(size),
            None => RtDim::InputAxis {
                tensor: 1,
                axis: rt_axis(counterpart_axis),
            },
        };
        current_dims.insert(axis, lead_dim.clone());
        let out_ty = TensorType {
            dims: current_dims.clone(),
            precision: ty.precision,
        };
        current = add_synth(
            dag,
            RiscOp::Expand {
                axis,
                size: size.clone(),
            },
            if matches!(size, RtDim::Lit(_)) {
                vec![current]
            } else {
                vec![current, counterpart]
            },
            out_ty,
            parent_span,
        );
    }

    for lead_axis in 0..lead_dims.len() {
        if current_dims[lead_axis] != lead_dims[lead_axis] && is_one_dim(&current_dims[lead_axis]) {
            current_dims[lead_axis] = lead_dims[lead_axis].clone();
            let out_ty = TensorType {
                dims: current_dims.clone(),
                precision: ty.precision,
            };
            let counterpart_lead_len = counterpart_ty.dims.len() - 2;
            let counterpart_axis = lead_axis
                .checked_sub(lead_dims.len().saturating_sub(counterpart_lead_len))
                .expect("a broadcast matmul lead axis must be supplied by the counterpart");
            let size = match dim_known_size(&lead_dims[lead_axis]) {
                Some(size) => RtDim::Lit(size),
                None => RtDim::InputAxis {
                    tensor: 1,
                    axis: rt_axis(counterpart_axis),
                },
            };
            current = add_synth(
                dag,
                RiscOp::Expand {
                    axis: lead_axis,
                    size: size.clone(),
                },
                if matches!(size, RtDim::Lit(_)) {
                    vec![current]
                } else {
                    vec![current, counterpart]
                },
                out_ty,
                parent_span,
            );
        }
    }

    debug_assert!(
        checked_matmul_dims_compatible(&current_dims[current_dims.len() - 2..], matrix_dims),
        "matmul matrix dims must already be checked before lowering: current={:?}, target={:?}",
        &current_dims[current_dims.len() - 2..],
        matrix_dims
    );
    current
}

fn checked_matmul_dims_compatible(current: &[DimInfo], target: &[DimInfo; 2]) -> bool {
    current.len() == target.len()
        && current
            .iter()
            .zip(target.iter())
            .all(|(current, target)| checked_dim_compatible(current, target))
}

fn checked_dim_compatible(current: &DimInfo, target: &DimInfo) -> bool {
    match (current, target) {
        (DimInfo::Lit(a), DimInfo::Lit(b)) => a == b,
        (DimInfo::Named(a_name, a_size), DimInfo::Named(b_name, b_size)) => {
            a_name == b_name
                || match (a_size, b_size) {
                    (Some(a), Some(b)) => a == b,
                    _ => true,
                }
        }
        (DimInfo::Named(_, Some(a)), DimInfo::Lit(b))
        | (DimInfo::Lit(b), DimInfo::Named(_, Some(a))) => a == b,
        (DimInfo::Named(_, None), DimInfo::Lit(_)) | (DimInfo::Lit(_), DimInfo::Named(_, None)) => {
            true
        }
    }
}

/// softmax(x, axis) = exp(x - max_reduce(x, axis)) / sum(exp(x - max_reduce(x, axis)), axis)
///
/// Lowering (spec §4.2): numerically stable softmax via max subtraction.
pub fn lower_softmax(
    dag: &mut Dag,
    x: NodeId,
    axis: usize,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    let red_ty = reduced_type(ty, axis);
    let axis_dim = require_dim(ty.dims.get(axis), "softmax axis");
    let size = match dim_known_size(&axis_dim) {
        Some(size) => RtDim::Lit(size),
        None => RtDim::InputAxis {
            tensor: 1,
            axis: rt_axis(axis),
        },
    };

    // 1. max_reduce(x, axis)
    let max_val = add_synth(
        dag,
        RiscOp::MaxReduce { axis },
        vec![x],
        red_ty.clone(),
        parent_span,
    );

    // 2. expand max back to original shape
    let max_expanded = add_synth(
        dag,
        RiscOp::Expand {
            axis,
            size: size.clone(),
        },
        if matches!(size, RtDim::Lit(_)) {
            vec![max_val]
        } else {
            vec![max_val, x]
        },
        ty.clone(),
        parent_span,
    );

    // 3. shifted = x - max (numerical stability)
    let shifted = lower_sub(dag, x, max_expanded, ty, parent_span);

    // 4. exp(shifted)
    let exp_shifted = add_synth(dag, RiscOp::Exp, vec![shifted], ty.clone(), parent_span);

    // 5. sum(exp, axis)
    let sum_exp = add_synth(
        dag,
        RiscOp::sum_default(axis, ty.precision)
            .expect("softmax operand precision should accept reduce_sum"),
        vec![exp_shifted],
        red_ty,
        parent_span,
    );

    // 6. expand sum back to original shape
    let sum_expanded = add_synth(
        dag,
        RiscOp::Expand {
            axis,
            size: size.clone(),
        },
        if matches!(size, RtDim::Lit(_)) {
            vec![sum_exp]
        } else {
            vec![sum_exp, x]
        },
        ty.clone(),
        parent_span,
    );

    // 7. exp / sum
    lower_div(dag, exp_shifted, sum_expanded, ty, parent_span)
}

/// mean(x, axis) = sum(x, axis) / dim_size
///
/// Lowering (spec §3.4): sum then divide by the axis size.
pub fn lower_mean(
    dag: &mut Dag,
    x: NodeId,
    axis: usize,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    let red_ty = reduced_type(ty, axis);

    // sum(x, axis)
    let sum_node = add_synth(
        dag,
        RiscOp::sum_default(axis, ty.precision)
            .expect("mean operand precision should accept reduce_sum"),
        vec![x],
        red_ty.clone(),
        parent_span,
    );

    // The divisor is the reduced-axis extent. When that extent is a
    // concrete literal the divisor is a compile-time `Const` (the simple
    // path). When the extent is RUNTIME-DERIVED (a `Named(_, None)` axis
    // produced by a `shrink`/`stride`/`reshape` window, per issue #320),
    // it is unknown at lowering time, so the divisor is built as a runtime
    // count: `sum(ones_like(x), axis)`. The ones tensor carries the same
    // (symbolic) operand type as `x`, so `sum` over `axis` yields the
    // runtime extent in the reduced shape. The symbolic dim is resolved at
    // eval time by `bind_symbolic_dims`, which infers the extent from the
    // input tensor's runtime shape — the `Const`-of-ones then gets its
    // concrete shape from the bound `output_type`. This carries the
    // operand's extent instead of demanding a concrete one in IR lowering.
    let divisor = match dim_size(ty, axis) {
        Some(dim_size_val) => {
            let count = add_synth(
                dag,
                RiscOp::synth_const(red_ty.precision, dim_size_val as f64),
                vec![],
                red_ty.clone(),
                parent_span,
            );
            // chelis#616: the count Const is shaped like the reduced sum but
            // has no input edge carrying that relation; record it as a
            // shape-dep so the C backend's anon-dim renaming ties the two
            // (instead of fragmenting the Const's wildcard dim into a fresh
            // sourceless `_anon_dim_*`).
            dag.add_shape_dep(count, sum_node);
            count
        }
        None => {
            // ones shaped exactly like the operand `x` (same symbolic dims).
            let ones = add_synth(
                dag,
                RiscOp::synth_const(ty.precision, 1.0),
                vec![],
                ty.clone(),
                parent_span,
            );
            dag.add_shape_dep(ones, x);
            // sum the ones over `axis` -> the runtime extent, reduced shape.
            add_synth(
                dag,
                RiscOp::sum_default(axis, ty.precision)
                    .expect("mean count precision should accept reduce_sum"),
                vec![ones],
                red_ty.clone(),
                parent_span,
            )
        }
    };

    // sum / extent
    lower_div(dag, sum_node, divisor, &red_ty, parent_span)
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
    parent_span: Option<&str>,
) -> NodeId {
    let axis = x_ty.dims.len().saturating_sub(1);
    let axis_size = RtDim::Lit(require_axis_size(x_ty, axis, "layer_norm"));
    let mean = lower_mean(dag, x, axis, x_ty, parent_span);
    let mean_expanded = add_synth(
        dag,
        RiscOp::Expand {
            axis,
            size: axis_size.clone(),
        },
        vec![mean],
        x_ty.clone(),
        parent_span,
    );
    let centered = lower_sub(dag, x, mean_expanded, x_ty, parent_span);
    let squared = add_synth(
        dag,
        RiscOp::Mul,
        vec![centered, centered],
        x_ty.clone(),
        parent_span,
    );
    let var = lower_mean(dag, squared, axis, x_ty, parent_span);
    let var_expanded = add_synth(
        dag,
        RiscOp::Expand {
            axis,
            size: axis_size,
        },
        vec![var],
        x_ty.clone(),
        parent_span,
    );
    let eps_const = add_synth(
        dag,
        RiscOp::synth_const(x_ty.precision, eps),
        vec![],
        x_ty.clone(),
        parent_span,
    );
    let denom_sq = add_synth(
        dag,
        RiscOp::Add,
        vec![var_expanded, eps_const],
        x_ty.clone(),
        parent_span,
    );
    let denom = add_synth(dag, RiscOp::Sqrt, vec![denom_sq], x_ty.clone(), parent_span);
    let normed = lower_div(dag, centered, denom, x_ty, parent_span);

    let gamma_node = expand_to_match(dag, gamma, gamma_ty.clone(), x, &x_ty.dims, parent_span);
    let beta_node = expand_to_match(dag, beta, beta_ty.clone(), x, &x_ty.dims, parent_span);
    let scaled = add_synth(
        dag,
        RiscOp::Mul,
        vec![normed, gamma_node],
        x_ty.clone(),
        parent_span,
    );
    add_synth(
        dag,
        RiscOp::Add,
        vec![scaled, beta_node],
        x_ty.clone(),
        parent_span,
    )
}

/// [05-OP-51] cross-correlation for any positive spatial rank. The gathered
/// window matrix orders its contraction axis as (channel, kernel axes...).
/// One matrix contraction owns accumulation across that entire axis.
#[allow(clippy::too_many_arguments)]
pub fn lower_conv(
    dag: &mut Dag,
    input: NodeId,
    kernel: NodeId,
    input_ty: &TensorType,
    kernel_ty: &TensorType,
    output_ty: &TensorType,
    strides: &[usize],
    padding: &[(usize, usize)],
    parent_span: Option<&str>,
) -> NodeId {
    let input_shape: Vec<_> = input_ty
        .dims
        .iter()
        .map(|d| require_dim_extent(d, "conv input axis"))
        .collect();
    let kernel_shape: Vec<_> = kernel_ty
        .dims
        .iter()
        .map(|d| require_dim_extent(d, "conv kernel axis"))
        .collect();
    assert!(
        input_shape.len() >= 3 && input_shape.len() == kernel_shape.len(),
        "conv requires equal ranks of at least 3"
    );
    let rank = input_shape.len() - 2;
    assert!(
        strides.len() == rank && padding.len() == rank,
        "conv requires one stride/padding entry per spatial axis"
    );
    assert_eq!(input_shape[1], kernel_shape[1], "conv channel mismatch");
    assert_eq!(
        input_ty.precision, kernel_ty.precision,
        "conv dtype mismatch"
    );
    assert!(input_ty.precision.is_float(), "conv requires float data");
    let checked = |n: Option<usize>| {
        n.filter(|n| i64::try_from(*n).is_ok())
            .expect("conv shape arithmetic exceeds int64")
    };
    let product = |dims: &[usize]| dims.iter().fold(1usize, |n, &d| checked(n.checked_mul(d)));
    let mut padded_shape = input_shape.clone();
    let mut output_shape = vec![input_shape[0], kernel_shape[0]];
    for axis in 0..rank {
        let (low, high) = padding[axis];
        let padded = checked(
            input_shape[axis + 2]
                .checked_add(low)
                .and_then(|n| n.checked_add(high)),
        );
        assert!(
            strides[axis] > 0 && kernel_shape[axis + 2] > 0 && kernel_shape[axis + 2] <= padded,
            "conv invalid kernel/stride/padding"
        );
        padded_shape[axis + 2] = padded;
        output_shape.push(checked(
            ((padded - kernel_shape[axis + 2]) / strides[axis]).checked_add(1),
        ));
    }
    assert_eq!(
        output_ty.dims.len(),
        output_shape.len(),
        "conv output rank mismatch"
    );
    for (expected, actual) in output_shape.iter().zip(&output_ty.dims) {
        if let DimInfo::Lit(actual) | DimInfo::Named(_, Some(actual)) = actual {
            assert_eq!(expected, actual, "conv output shape mismatch");
        }
    }
    let precision = input_ty.precision;
    let ty = |dims: &[usize], precision| TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    };
    let mut pad = vec![(RtDim::Lit(0), RtDim::Lit(0)); 2];
    pad.extend(
        padding
            .iter()
            .map(|&(low, high)| (RtDim::Lit(low), RtDim::Lit(high))),
    );
    let padded = add_synth(
        dag,
        RiscOp::zero_pad(precision, pad),
        vec![input],
        ty(&padded_shape, precision),
        parent_span,
    );
    let padded_len = product(&padded_shape);
    let flat = add_synth(
        dag,
        RiscOp::Reshape {
            new_shape: vec![RtDim::Lit(padded_len)],
        },
        vec![padded],
        ty(&[padded_len], precision),
        parent_span,
    );
    let kernel_volume = product(&kernel_shape[2..]);
    let contracted = checked(input_shape[1].checked_mul(kernel_volume));
    let output_volume = product(&output_shape[2..]);
    let columns = checked(input_shape[0].checked_mul(output_volume));
    let index_count = checked(contracted.checked_mul(columns));
    let mut indices = Vec::with_capacity(index_count);
    for contraction in 0..contracted {
        let channel = contraction / kernel_volume;
        let mut kernel_position = contraction % kernel_volume;
        let mut offsets = vec![0; rank];
        for axis in (0..rank).rev() {
            offsets[axis] = kernel_position % kernel_shape[axis + 2];
            kernel_position /= kernel_shape[axis + 2];
        }
        for column in 0..columns {
            let batch = column / output_volume;
            let mut output_position = column % output_volume;
            let mut coordinates = vec![0; rank];
            for axis in (0..rank).rev() {
                coordinates[axis] = output_position % output_shape[axis + 2];
                output_position /= output_shape[axis + 2];
            }
            let mut index = checked(
                batch
                    .checked_mul(input_shape[1])
                    .and_then(|n| n.checked_add(channel)),
            );
            for axis in 0..rank {
                let coordinate = checked(
                    coordinates[axis]
                        .checked_mul(strides[axis])
                        .and_then(|n| n.checked_add(offsets[axis])),
                );
                index = checked(
                    index
                        .checked_mul(padded_shape[axis + 2])
                        .and_then(|n| n.checked_add(coordinate)),
                );
            }
            indices.push(
                chelis_types::scalar_from_i64(
                    "conv index",
                    Prim::Int64,
                    i64::try_from(index).expect("checked index"),
                )
                .expect("int64 index"),
            );
        }
    }
    let matrix_shape = [contracted, columns];
    let index_node = add_synth(
        dag,
        RiscOp::ConstTensor {
            data: chelis_types::tensor_from_scalars(Prim::Int64, &indices),
        },
        vec![],
        ty(&matrix_shape, Prim::Int64),
        parent_span,
    );
    let windows = add_synth(
        dag,
        RiscOp::Gather { axis: 0 },
        vec![flat, index_node],
        ty(&matrix_shape, precision),
        parent_span,
    );
    let kernel_matrix_shape = [kernel_shape[0], contracted];
    let kernel_matrix = add_synth(
        dag,
        RiscOp::Reshape {
            new_shape: kernel_matrix_shape.iter().map(|&n| RtDim::Lit(n)).collect(),
        },
        vec![kernel],
        ty(&kernel_matrix_shape, precision),
        parent_span,
    );
    let contracted_node = lower_matmul(
        dag,
        kernel_matrix,
        windows,
        &ty(&kernel_matrix_shape, precision),
        &ty(&matrix_shape, precision),
        parent_span,
    );
    let mut reshaped = vec![kernel_shape[0], input_shape[0]];
    reshaped.extend(&output_shape[2..]);
    let result = add_synth(
        dag,
        RiscOp::Reshape {
            new_shape: reshaped.iter().map(|&n| RtDim::Lit(n)).collect(),
        },
        vec![contracted_node],
        ty(&reshaped, precision),
        parent_span,
    );
    let mut axes: Vec<_> = (0..output_shape.len()).collect();
    axes.swap(0, 1);
    add_synth(
        dag,
        RiscOp::Permute { axes },
        vec![result],
        ty(&output_shape, precision),
        parent_span,
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
    fn sub_produces_direct_identity() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 5.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let result = lower_sub(&mut dag, a, b, &scalar_f32(), None);
        assert!(verify::verify(&dag).is_empty());

        assert_eq!(dag.len(), 3);
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Sub);
        assert_eq!(result_node.inputs, vec![a, b]);
    }

    #[test]
    fn relu_produces_dedicated_identity() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, -1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let result = lower_relu(&mut dag, x, &scalar_f32(), None);
        assert!(verify::verify(&dag).is_empty());

        assert_eq!(dag.len(), 2);
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Relu);
        assert_eq!(result_node.inputs, vec![x]);
    }

    #[test]
    fn sigmoid_produces_correct_chain() {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let result = lower_sigmoid(&mut dag, x, &scalar_f32(), None);
        assert!(verify::verify(&dag).is_empty());

        // sigmoid(x) = recip(1 + exp(-x)). Chain: x, neg(x),
        // exp(neg(x)), const(1), add, recip = 6 nodes; ends in Recip.
        assert_eq!(dag.len(), 6);
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Recip);

        // Tighten the structural pin so a future refactor can't quietly
        // re-introduce a `Log` step or drop one of the inner ops while
        // still ending at `Recip`.
        let ops: Vec<&RiscOp> = dag.nodes().iter().map(|n| &n.op).collect();
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Neg)),
            "sigmoid must contain Neg(x)"
        );
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Exp)),
            "sigmoid must contain Exp(neg(x))"
        );
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Add)),
            "sigmoid must contain Add(const(1), exp(neg(x)))"
        );
        assert_eq!(
            ops.iter().filter(|op| matches!(op, RiscOp::Recip)).count(),
            1,
            "sigmoid must contain exactly one Recip"
        );
        assert!(
            ops.iter().all(|op| !matches!(op, RiscOp::Log)),
            "sigmoid lowering must not contain Log"
        );

        // Verify the Recip's input is the Add node (the structural
        // pin the prior one-way root-op check would have missed).
        let recip_node = dag.get(result).unwrap();
        let recip_input = dag.get(recip_node.inputs[0]).unwrap();
        assert!(
            matches!(recip_input.op, RiscOp::Add),
            "Recip must consume the Add node directly, got {:?}",
            recip_input.op
        );
    }

    #[test]
    fn lower_div_emits_single_div_node() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 6.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let result = lower_div(&mut dag, a, b, &scalar_f32(), None);
        assert!(verify::verify(&dag).is_empty());

        // lower_div emits a single `RiscOp::Div(a, b)` node
        // (was 4 ops: log, neg, exp, mul). Chain: a, b, div = 3 nodes.
        assert_eq!(dag.len(), 3);
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Div);
        assert_eq!(result_node.inputs, vec![a, b]);
        // Regression guard (#172 div-via-recip parity): `div` must lower
        // to NATIVE IEEE `RiscOp::Div`, never `a * recip(b)` nor the
        // `a * exp(neg(log(b)))` recip-via-log decomposition. The
        // mul/recip form differs from torch's direct division by ~1 ULP
        // on ~30% of f32 operand pairs, and the log form NaNs on
        // non-positive divisors. None of Log/Exp/Neg/Recip/Mul should
        // appear in a bare `div` lowering.
        assert!(
            dag.nodes().iter().all(|n| !matches!(
                n.op,
                RiscOp::Log | RiscOp::Exp | RiscOp::Neg | RiscOp::Recip | RiscOp::Mul
            )),
            "div lowering must be native Div, not a recip/log/mul decomposition (#172)"
        );
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
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 5.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let result = lower_gt(&mut dag, a, b, &scalar_f32(), None);

        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        // b, a order (swapped).
        assert_eq!(node.inputs, vec![b, a]);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn gte_produces_not_cmplt() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 5.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let result = lower_gte(&mut dag, a, b, &scalar_f32(), None);

        // a, b, CmpLt(a,b), Const(1), CmpLt(lt, 1)
        assert_eq!(dag.len(), 5);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn lte_produces_not_cmplt_ba() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 5.0),
            vec![],
            scalar_f32(),
            None,
        );
        let result = lower_lte(&mut dag, a, b, &scalar_f32(), None);

        assert_eq!(dag.len(), 5);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn eq_produces_not_or_cmplt() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let result = lower_eq(&mut dag, a, b, &scalar_f32(), None);

        // a, b, CmpLt(a,b), CmpLt(b,a), MaxElem, Const(1), CmpLt(or, 1)
        assert_eq!(dag.len(), 7);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn neq_produces_or_cmplt() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 5.0),
            vec![],
            scalar_f32(),
            None,
        );
        let result = lower_neq(&mut dag, a, b, &scalar_f32(), None);

        // a, b, CmpLt(a,b), CmpLt(b,a), MaxElem
        assert_eq!(dag.len(), 5);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::MaxElem);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn min_elem_produces_direct_selection() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 5.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let result = lower_min_elem(&mut dag, a, b, &scalar_f32(), None);
        assert!(verify::verify(&dag).is_empty());

        assert_eq!(dag.len(), 3);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::MinElem);
        assert_eq!(node.inputs, vec![a, b]);
    }

    // --- H2: Boolean operators ---

    #[test]
    fn and_produces_mul() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_bool().precision, 1.0),
            vec![],
            scalar_bool(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_bool().precision, 0.0),
            vec![],
            scalar_bool(),
            None,
        );
        let result = lower_and(&mut dag, a, b, &scalar_bool(), None);

        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::Mul);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn or_produces_max_elem() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_bool().precision, 0.0),
            vec![],
            scalar_bool(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_bool().precision, 1.0),
            vec![],
            scalar_bool(),
            None,
        );
        let result = lower_or(&mut dag, a, b, &scalar_bool(), None);

        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::MaxElem);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn not_produces_cmplt_with_one() {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_bool().precision, 1.0),
            vec![],
            scalar_bool(),
            None,
        );
        let result = lower_not(&mut dag, a, &scalar_bool(), None);

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
        let a = dag.add_node(
            RiscOp::Load { name: "A".into() },
            vec![],
            a_ty.clone(),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "B".into() },
            vec![],
            b_ty.clone(),
            None,
        );
        let result = lower_matmul(&mut dag, a, b, &a_ty, &b_ty, None);

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
            ops.iter()
                .any(|op| matches!(op, RiscOp::Sum { axis: 1, .. })),
            "expected a Sum{{axis:1}} node"
        );

        // Result type should be [2, 4]
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.output_type.dims.len(), 2);
        assert_eq!(result_node.output_type.dims[0], DimInfo::Lit(2));
        assert_eq!(result_node.output_type.dims[1], DimInfo::Lit(4));
    }

    #[test]
    fn batched_matmul_rank4_produces_batched_sum_axis() {
        let mut dag = Dag::new();
        let a_ty = TensorType {
            dims: vec![
                DimInfo::Lit(2),
                DimInfo::Lit(3),
                DimInfo::Lit(5),
                DimInfo::Lit(7),
            ],
            precision: Prim::F32,
        };
        let b_ty = TensorType {
            dims: vec![
                DimInfo::Lit(2),
                DimInfo::Lit(3),
                DimInfo::Lit(7),
                DimInfo::Lit(11),
            ],
            precision: Prim::F32,
        };
        let a = dag.add_node(
            RiscOp::Load { name: "A".into() },
            vec![],
            a_ty.clone(),
            None,
        );
        let b = dag.add_node(
            RiscOp::Load { name: "B".into() },
            vec![],
            b_ty.clone(),
            None,
        );
        let result = lower_matmul(&mut dag, a, b, &a_ty, &b_ty, None);
        let result_node = dag.get(result).unwrap();
        assert_eq!(
            result_node.output_type.dims,
            vec![
                DimInfo::Lit(2),
                DimInfo::Lit(3),
                DimInfo::Lit(5),
                DimInfo::Lit(11)
            ]
        );
        assert!(matches!(result_node.op, RiscOp::Sum { axis: 3, .. }));
    }

    #[test]
    fn softmax_produces_maxreduce_sub_exp_sum_div() {
        let mut dag = Dag::new();
        let ty = vec_5();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
        let result = lower_softmax(&mut dag, x, 0, &ty, None);

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
            ops.iter()
                .any(|op| matches!(op, RiscOp::Sum { axis: 0, .. })),
            "expected Sum"
        );
        // Sub remains its own primitive identity.
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Sub)),
            "expected direct Sub"
        );

        // the final result is now a single `Div` node
        // (was `Mul` from the recip-via-log decomposition).
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Div);
        // Regression guard: the lowering must not introduce a Log
        // (used to come from the div-via-recip-via-log decomposition).
        assert!(
            ops.iter().all(|op| !matches!(op, RiscOp::Log)),
            "softmax lowering must not contain Log"
        );
    }

    #[test]
    fn mean_produces_sum_div() {
        let mut dag = Dag::new();
        let ty = vec_5();
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
        let result = lower_mean(&mut dag, x, 0, &ty, None);

        let ops: Vec<_> = dag.nodes().iter().map(|n| &n.op).collect();
        // Should have Sum
        assert!(
            ops.iter()
                .any(|op| matches!(op, RiscOp::Sum { axis: 0, .. })),
            "expected Sum"
        );
        // Should have Const(5.0) for dim size
        assert!(
            ops.iter()
                .any(|op| matches!(op, RiscOp::Const { value } if value.as_f64_lossy() == 5.0)),
            "expected Const(5.0) for dimension size"
        );
        // mean is now `Div(sum, const(dim_size))` — a
        // single `Div` node, not the prior `mul(sum, exp(neg(log(_))))`
        // chain.
        let result_node = dag.get(result).unwrap();
        assert_eq!(result_node.op, RiscOp::Div);
        // Regression guard: no Log / Exp / Neg from the old div lowering.
        assert!(
            ops.iter()
                .all(|op| !matches!(op, RiscOp::Log | RiscOp::Exp | RiscOp::Neg)),
            "mean lowering must not contain Log/Exp/Neg"
        );
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
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            x_ty.clone(),
            None,
        );
        let gamma = dag.add_node(
            RiscOp::Load {
                name: "gamma".into(),
            },
            vec![],
            scale_ty.clone(),
            None,
        );
        let beta = dag.add_node(
            RiscOp::Load {
                name: "beta".into(),
            },
            vec![],
            scale_ty.clone(),
            None,
        );
        let result = lower_layer_norm(
            &mut dag, x, gamma, beta, &x_ty, &scale_ty, &scale_ty, 1e-5, None,
        );

        let ops: Vec<_> = dag.nodes().iter().map(|n| &n.op).collect();
        assert!(
            ops.iter()
                .any(|op| matches!(op, RiscOp::Sum { axis: 1, .. })),
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
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            x_ty.clone(),
            None,
        );
        let gamma = dag.add_node(
            RiscOp::Load {
                name: "gamma".into(),
            },
            vec![],
            scale_ty.clone(),
            None,
        );
        let beta = dag.add_node(
            RiscOp::Load {
                name: "beta".into(),
            },
            vec![],
            scale_ty.clone(),
            None,
        );
        let _ = lower_layer_norm(
            &mut dag, x, gamma, beta, &x_ty, &scale_ty, &scale_ty, 1e-5, None,
        );
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
    fn conv_produces_im2col_style_sequence() {
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
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            input_ty.clone(),
            None,
        );
        let kernel = dag.add_node(
            RiscOp::Load { name: "w".into() },
            vec![],
            kernel_ty.clone(),
            None,
        );
        let result = lower_conv(
            &mut dag,
            input,
            kernel,
            &input_ty,
            &kernel_ty,
            &output_ty,
            &[1, 1],
            &[(0, 0), (0, 0)],
            None,
        );

        let ops: Vec<_> = dag.nodes().iter().map(|n| &n.op).collect();
        assert!(ops.iter().any(|op| matches!(op, RiscOp::Pad { .. })));
        assert!(ops.iter().any(|op| matches!(op, RiscOp::Gather { .. })));
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
        let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
        let out = lower_softmax(&mut dag, x, 0, &ty, None);
        let expand_sizes: Vec<_> = dag
            .nodes()
            .iter()
            .filter_map(|node| match &node.op {
                RiscOp::Expand { size, .. } => Some(size.clone()),
                _ => None,
            })
            .collect();

        assert!(
            expand_sizes.iter().all(|size| matches!(
                size,
                RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0)
                }
            )),
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
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            x_ty.clone(),
            None,
        );
        let gamma = dag.add_node(
            RiscOp::Load {
                name: "gamma".into(),
            },
            vec![],
            scale_ty.clone(),
            None,
        );
        let beta = dag.add_node(
            RiscOp::Load {
                name: "beta".into(),
            },
            vec![],
            scale_ty.clone(),
            None,
        );
        let _ = lower_layer_norm(
            &mut dag, x, gamma, beta, &x_ty, &scale_ty, &scale_ty, 1e-5, None,
        );
    }

    #[test]
    #[should_panic(expected = "conv output rank mismatch")]
    fn conv_rejects_missing_output_shape() {
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
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            input_ty.clone(),
            None,
        );
        let kernel = dag.add_node(
            RiscOp::Load { name: "w".into() },
            vec![],
            kernel_ty.clone(),
            None,
        );
        let _ = lower_conv(
            &mut dag,
            input,
            kernel,
            &input_ty,
            &kernel_ty,
            &output_ty,
            &[1, 1],
            &[(1, 1), (1, 1)],
            None,
        );
    }
}
