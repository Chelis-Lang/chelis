//! Tier 2 decomposition helpers.
//!
//! These functions decompose Tier 2 (derived) operations into Tier 1 RISC DAG nodes.
//!
//! Span propagation per `spec/design/chelis_span_survival.md` §2.3 Tier 2
//! row: every synthesized sub-node inherits the decomposed parent's
//! `span_id`. If the parent had no span, sub-nodes carry the canonical
//! `__synthesized_tier2__` marker (defined in
//! `spec/03-deep-syntax.md` §1.1.1).
//!
//! Each public lowerer takes `parent_span: Option<&str>`:
//!   * `Some(s)` — the operation's source span; sub-nodes inherit `s`.
//!   * `None` — no source region; sub-nodes carry `__synthesized_tier2__`.
//!
//! The internal `add_synth` helper applies the rule once per node so we
//! don't duplicate it across the ~109 `add_node` callsites.

use chelis_types::types::Prim;

use crate::dag::{Dag, DimExpr, DimInfo, NodeId, RiscOp, RtDim, TensorType};

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

/// `sub(a, b)` = `add(a, neg(b))`
pub fn lower_sub(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    let neg_b = add_synth(dag, RiscOp::Neg, vec![b], ty.clone(), parent_span);
    add_synth(dag, RiscOp::Add, vec![a, neg_b], ty.clone(), parent_span)
}

/// `relu(x)` = `max_elem(x, const(0))`
pub fn lower_relu(dag: &mut Dag, x: NodeId, ty: &TensorType, parent_span: Option<&str>) -> NodeId {
    let zero = add_synth(
        dag,
        RiscOp::Const { value: 0.0 },
        vec![],
        ty.clone(),
        parent_span,
    );
    add_synth(dag, RiscOp::MaxElem, vec![x, zero], ty.clone(), parent_span)
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
        RiscOp::Const { value: 1.0 },
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
        RiscOp::Const { value: 2.0 },
        vec![],
        ty.clone(),
        parent_span,
    );
    let two_x = add_synth(dag, RiscOp::Mul, vec![two, x], ty.clone(), parent_span);
    let sig_2x = lower_sigmoid(dag, two_x, ty, parent_span);
    let two_again = add_synth(
        dag,
        RiscOp::Const { value: 2.0 },
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
        RiscOp::Const { value: -1.0 },
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
        RiscOp::Const {
            value: 0.7978845608028654,
        },
        vec![],
        ty.clone(),
        parent_span,
    );
    let k = add_synth(
        dag,
        RiscOp::Const { value: 0.044715 },
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
        RiscOp::Const { value: 1.0 },
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
        RiscOp::Const { value: 0.5 },
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
        RiscOp::Const { value: 1.0 },
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
        RiscOp::Const { value: 1.0 },
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
        RiscOp::Const { value: 1.0 },
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

/// H1: `min_elem(a, b)` = `neg(max_elem(neg(a), neg(b)))`
pub fn lower_min_elem(
    dag: &mut Dag,
    a: NodeId,
    b: NodeId,
    ty: &TensorType,
    parent_span: Option<&str>,
) -> NodeId {
    let neg_a = add_synth(dag, RiscOp::Neg, vec![a], ty.clone(), parent_span);
    let neg_b = add_synth(dag, RiscOp::Neg, vec![b], ty.clone(), parent_span);
    let max = add_synth(
        dag,
        RiscOp::MaxElem,
        vec![neg_a, neg_b],
        ty.clone(),
        parent_span,
    );
    add_synth(dag, RiscOp::Neg, vec![max], ty.clone(), parent_span)
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
        RiscOp::Const { value: 1.0 },
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
    target_dims: &[DimInfo],
    parent_span: Option<&str>,
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
        node = add_synth(
            dag,
            RiscOp::Expand {
                axis: insert_at,
                size,
            },
            vec![node],
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
    let i_size = DimExpr::from(&i_dim);
    let k_size = DimExpr::from(&k_dim);

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
        &lead_dims,
        &[i_dim.clone(), j_dim.clone()],
        parent_span,
    );
    let a_expanded = add_synth(
        dag,
        RiscOp::Expand {
            axis: lead_len + 2,
            size: k_size,
        },
        vec![a_aligned],
        expanded_ty.clone(),
        parent_span,
    );

    let b_aligned = align_matmul_operand(
        dag,
        b,
        b_ty,
        &lead_dims,
        &[j_dim.clone(), k_dim.clone()],
        parent_span,
    );
    let b_with_i = add_synth(
        dag,
        RiscOp::Expand {
            axis: lead_len,
            size: i_size,
        },
        vec![b_aligned],
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
    lead_dims: &[DimInfo],
    matrix_dims: &[DimInfo; 2],
    parent_span: Option<&str>,
) -> NodeId {
    let source_lead = &ty.dims[..ty.dims.len() - 2];
    let mut current = node;
    let mut current_dims = ty.dims.clone();
    let missing = lead_dims.len().saturating_sub(source_lead.len());
    for (axis, lead_dim) in lead_dims.iter().take(missing).enumerate() {
        let size = DimExpr::from(lead_dim);
        current_dims.insert(axis, lead_dim.clone());
        let out_ty = TensorType {
            dims: current_dims.clone(),
            precision: ty.precision,
        };
        current = add_synth(
            dag,
            RiscOp::Expand { axis, size },
            vec![current],
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
            current = add_synth(
                dag,
                RiscOp::Expand {
                    axis: lead_axis,
                    size: DimExpr::from(&lead_dims[lead_axis]),
                },
                vec![current],
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
    let size = DimExpr::from(&require_dim(ty.dims.get(axis), "softmax axis"));

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
        vec![max_val],
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
        RiscOp::Expand { axis, size },
        vec![sum_exp],
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
        Some(dim_size_val) => add_synth(
            dag,
            RiscOp::Const {
                value: dim_size_val as f64,
            },
            vec![],
            red_ty.clone(),
            parent_span,
        ),
        None => {
            // ones shaped exactly like the operand `x` (same symbolic dims).
            let ones = add_synth(
                dag,
                RiscOp::Const { value: 1.0 },
                vec![],
                ty.clone(),
                parent_span,
            );
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
    let axis_size = DimExpr::Concrete(require_axis_size(x_ty, axis, "layer_norm"));
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
        RiscOp::Const { value: eps },
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

    let gamma_node = expand_to_match(dag, gamma, gamma_ty.clone(), &x_ty.dims, parent_span);
    let beta_node = expand_to_match(dag, beta, beta_ty.clone(), &x_ty.dims, parent_span);
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
    parent_span: Option<&str>,
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
            "IR conv2d kernel dims ({kh_size}, {kw_size}) exceed padded input dims ({padded_h}, {padded_w})"
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
            "IR conv2d output shape mismatch: expected spatial dims ({strided_h}, {strided_w}), got ({h_out_size}, {w_out_size})"
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
    let padded = add_synth(
        dag,
        RiscOp::Pad {
            padding: vec![
                (RtDim::Lit(0), RtDim::Lit(0)),
                (RtDim::Lit(0), RtDim::Lit(0)),
                (RtDim::Lit(padding), RtDim::Lit(padding)),
                (RtDim::Lit(padding), RtDim::Lit(padding)),
            ],
            fill: 0.0,
        },
        vec![input],
        padded_ty.clone(),
        parent_span,
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
                parent_span,
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
                parent_span,
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
                parent_span,
            );

            acc = Some(match acc {
                Some(prev) => add_synth(
                    dag,
                    RiscOp::Add,
                    vec![prev, term],
                    output_ty.clone(),
                    parent_span,
                ),
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
    parent_span: Option<&str>,
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
    let sampled_window = add_synth(
        dag,
        RiscOp::Shrink {
            bounds: vec![
                (
                    RtDim::Lit(0),
                    RtDim::Lit(require_dim_extent(batch, "conv2d batch axis")),
                ),
                (
                    RtDim::Lit(0),
                    RtDim::Lit(require_dim_extent(in_c, "conv2d input channel axis")),
                ),
                (RtDim::Lit(kh_idx), RtDim::Lit(kh_idx + sample_h)),
                (RtDim::Lit(kw_idx), RtDim::Lit(kw_idx + sample_w)),
            ],
        },
        vec![padded],
        sampled_window_ty.clone(),
        parent_span,
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
    add_synth(
        dag,
        RiscOp::Stride {
            strides: vec![
                RtDim::Lit(1),
                RtDim::Lit(1),
                RtDim::Lit(stride),
                RtDim::Lit(stride),
            ],
        },
        vec![sampled_window],
        sampled_ty,
        parent_span,
    )
}

#[allow(clippy::too_many_arguments)]
fn lower_conv2d_kernel_slice(
    dag: &mut Dag,
    kernel: NodeId,
    out_c: &DimInfo,
    in_c: &DimInfo,
    kh_idx: usize,
    kw_idx: usize,
    precision: Prim,
    parent_span: Option<&str>,
) -> NodeId {
    add_synth(
        dag,
        RiscOp::Shrink {
            bounds: vec![
                (
                    RtDim::Lit(0),
                    RtDim::Lit(require_dim_extent(out_c, "conv2d output channel axis")),
                ),
                (
                    RtDim::Lit(0),
                    RtDim::Lit(require_dim_extent(in_c, "conv2d input channel axis")),
                ),
                (RtDim::Lit(kh_idx), RtDim::Lit(kh_idx + 1)),
                (RtDim::Lit(kw_idx), RtDim::Lit(kw_idx + 1)),
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
        parent_span,
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
    parent_span: Option<&str>,
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
    let permuted = add_synth(
        dag,
        RiscOp::Permute {
            axes: vec![1, 0, 2, 3],
        },
        vec![input],
        permuted_ty.clone(),
        parent_span,
    );

    let h_out_size = require_dim_extent(&h_out, "conv2d output height axis");
    let w_out_size = require_dim_extent(&w_out, "conv2d output width axis");
    let col_dim = DimInfo::Lit(batch_size * h_out_size * w_out_size);
    let cols_ty = TensorType {
        dims: vec![input_ty.dims[1].clone(), col_dim.clone()],
        precision: input_ty.precision,
    };
    let cols = add_synth(
        dag,
        RiscOp::Reshape {
            new_shape: cols_ty.dims.clone(),
        },
        vec![permuted],
        cols_ty.clone(),
        parent_span,
    );

    let kernel_flat_ty = TensorType {
        dims: vec![out_c.clone(), DimInfo::Lit(in_c_size)],
        precision: kernel_ty.precision,
    };
    let kernel_flat = add_synth(
        dag,
        RiscOp::Reshape {
            new_shape: kernel_flat_ty.dims.clone(),
        },
        vec![kernel],
        kernel_flat_ty.clone(),
        parent_span,
    );

    let product = lower_matmul(
        dag,
        kernel_flat,
        cols,
        &kernel_flat_ty,
        &cols_ty,
        parent_span,
    );
    let product_4d_ty = TensorType {
        dims: vec![
            out_c.clone(),
            input_ty.dims[0].clone(),
            h_out.clone(),
            w_out.clone(),
        ],
        precision: output_ty.precision,
    };
    let product_4d = add_synth(
        dag,
        RiscOp::Reshape {
            new_shape: product_4d_ty.dims.clone(),
        },
        vec![product],
        product_4d_ty,
        parent_span,
    );
    add_synth(
        dag,
        RiscOp::Permute {
            axes: vec![1, 0, 2, 3],
        },
        vec![product_4d],
        output_ty.clone(),
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
    fn sub_produces_add_neg() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
        let result = lower_sub(&mut dag, a, b, &scalar_f32(), None);
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
        let x = dag.add_node(RiscOp::Const { value: -1.0 }, vec![], scalar_f32(), None);
        let result = lower_relu(&mut dag, x, &scalar_f32(), None);
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
        let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
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
        let a = dag.add_node(RiscOp::Const { value: 6.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
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
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
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
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
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
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32(), None);
        let result = lower_lte(&mut dag, a, b, &scalar_f32(), None);

        assert_eq!(dag.len(), 5);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn eq_produces_not_or_cmplt() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
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
        let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32(), None);
        let result = lower_neq(&mut dag, a, b, &scalar_f32(), None);

        // a, b, CmpLt(a,b), CmpLt(b,a), MaxElem
        assert_eq!(dag.len(), 5);
        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::MaxElem);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn min_elem_produces_neg_max_neg() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32(), None);
        let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
        let result = lower_min_elem(&mut dag, a, b, &scalar_f32(), None);
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
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_bool(), None);
        let b = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_bool(), None);
        let result = lower_and(&mut dag, a, b, &scalar_bool(), None);

        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::Mul);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn or_produces_max_elem() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_bool(), None);
        let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_bool(), None);
        let result = lower_or(&mut dag, a, b, &scalar_bool(), None);

        let node = dag.get(result).unwrap();
        assert_eq!(node.op, RiscOp::MaxElem);
        assert_eq!(node.output_type.precision, Prim::Bool);
    }

    #[test]
    fn not_produces_cmplt_with_one() {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_bool(), None);
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
        // Sub still produces Add+Neg.
        assert!(
            ops.iter().any(|op| matches!(op, RiscOp::Neg)),
            "expected Neg (from sub)"
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
                .any(|op| matches!(op, RiscOp::Const { value } if *value == 5.0)),
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
        let result = lower_conv2d(
            &mut dag, input, kernel, &input_ty, &kernel_ty, &output_ty, 1, 0, None,
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
        let _ = lower_conv2d(
            &mut dag, input, kernel, &input_ty, &kernel_ty, &output_ty, 1, 1, None,
        );
    }
}
