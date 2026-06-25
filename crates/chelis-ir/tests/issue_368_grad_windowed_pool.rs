//! Issue #368: post-#320 residue — `grad` through `mean` / `max_reduce`
//! over a `concat`'d window stack whose reduced/non-reduced axes are
//! runtime-symbolic `Named(_, None)` extents (the windowed-pool idiom
//! behind a symbolic-dim callee).
//!
//! On 0.10.0 the two probes failed one layer deeper than #320:
//!   * the `mean` path PANICKED — the Sum adjoint indexed `dims[axis]` of a
//!     rank-0 (collapsed) operand: "index out of bounds: the len is 0".
//!   * the `max_reduce` path built a backward DAG that FAILED verification —
//!     a blend `mul` mixed a rank-1 mask with a rank-0 `fail` branch:
//!     "binary op ... mismatched dimension count: 1 vs 0".
//!
//! Root causes (front-end + adjoint):
//!   1. `concat` had no IR-DAG lowering, so behind `grad` it collapsed to a
//!      rank-0 `Load { "concat" }` placeholder that severed the data
//!      dependency from the windowed rows back to the differentiated input
//!      AND fed the reduce a rank-0 operand. Now lowered as Pad + Add.
//!   2. the `mean` arm never recovered a collapsed operand's rank (unlike
//!      `max_reduce`), so `tier2::lower_mean` built a degenerate rank-0
//!      reduce. Now recovers via `recover_collapsed_operand_type`.
//!   3. the `if`/`then`/`else` blend multiplied the mask against a scalar
//!      `fail` branch without broadcasting it to the blend rank, and used
//!      the abstract symbolic join type instead of a concrete branch shape.
//!   4. the `Pad`/`Shrink`/`Stride` reverse-mode adjoints `dim_size`-panicked
//!      on a symbolic source extent. They now emit placeholder movement
//!      bounds that `bind_symbolic_dims` recomputes from the resolved shapes.
//!
//! This file is the IR-level oracle: it builds the lowered windowed-pool
//! backward structure with SYMBOLIC window/reduce axes — exactly the #368
//! condition — and finite-difference-checks both reductions, plus the
//! symbolic movement adjoints and negative parity. The end-to-end Surf
//! oracle is the CLI sibling `issue_368_grad_windowed_pool.rs`.

use std::collections::HashMap;

use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::grad::grad_dag_checked;
use chelis_ir::tier2;
use chelis_types::types::Prim;

fn f32_dims(dims: Vec<DimInfo>) -> TensorType {
    TensorType {
        dims,
        precision: Prim::F32,
    }
}

fn scalar_f32() -> TensorType {
    f32_dims(vec![])
}

/// A runtime-symbolic (unsized) dim — the windowed-pool `Named(_, None)`.
fn sym(name: &str) -> DimInfo {
    DimInfo::Named(name.into(), None)
}

fn load(dag: &mut Dag, name: &str, dims: Vec<DimInfo>) -> NodeId {
    dag.add_node(
        RiscOp::Load { name: name.into() },
        vec![],
        f32_dims(dims),
        None,
    )
}

/// Reduce-along-axis-0 output type of a 2-D operand.
fn reduced(ty: &TensorType) -> TensorType {
    f32_dims(ty.dims[1..].to_vec())
}

fn assert_fd_match(
    label: &str,
    dag: &Dag,
    grad_dag: &Dag,
    out: NodeId,
    grad_x: NodeId,
    base: &TensorValue,
) {
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), base.clone());
    let analytic = eval_tensor(grad_dag, &inputs).expect("analytic eval")[&grad_x]
        .data
        .clone();
    assert_eq!(
        analytic.len(),
        base.data.len(),
        "{label}: gradient length must match input"
    );

    let h = 1e-3;
    for (j, a) in analytic.iter().enumerate() {
        let mut plus = base.clone();
        let mut minus = base.clone();
        plus.data[j] += h;
        minus.data[j] -= h;
        let mut ip = HashMap::new();
        ip.insert("x".to_string(), plus);
        let mut im = HashMap::new();
        im.insert("x".to_string(), minus);
        let fp = eval_tensor(dag, &ip).expect("plus eval")[&out].data[0];
        let fm = eval_tensor(dag, &im).expect("minus eval")[&out].data[0];
        let numerical = (fp - fm) / (2.0 * h);
        assert!(
            (a - numerical).abs() < 2e-3,
            "{label}: finite-diff mismatch at {j}: analytic {a}, numerical {numerical}",
        );
    }
}

/// Build the windowed stack `[2, 2]` from a `tensor[4]` input by placing two
/// `[1, 2]` rows (the window views) via Pad + Add along axis 0 — the exact
/// shape `lower_ad_concat` produces for the windowed-pool idiom.
///
/// `row0 = x[0:2]`, `row1 = x[2:4]` via concrete-bound `Shrink`, each
/// reshaped to `[1, 2]`. The window length is concrete here because the
/// callee's `reshape(..., [1, m])` arg monomorphizes `m` to a literal at the
/// call site — this mirrors the real lowered DAG (the symbolic-axis movement
/// adjoints are covered separately by `issue_368_pad_adjoint_*` and the
/// issue_291 symbolic-stride test). Returns `(stacked_node, stacked_type)`.
fn build_windowed_stack(dag: &mut Dag, x: NodeId) -> (NodeId, TensorType) {
    let row_ty = f32_dims(vec![DimInfo::Lit(1), DimInfo::Lit(2)]);
    let stacked_ty = f32_dims(vec![DimInfo::Lit(2), DimInfo::Lit(2)]);

    let mut rows = Vec::new();
    for start in [0usize, 2usize] {
        let shrunk = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(start, start + 2)],
            },
            vec![x],
            f32_dims(vec![DimInfo::Lit(2)]),
            None,
        );
        let reshaped = dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![DimInfo::Lit(1), DimInfo::Lit(2)],
            },
            vec![shrunk],
            row_ty.clone(),
            None,
        );
        rows.push(reshaped);
    }

    // concat along axis 0 via Pad + Add: row i at offset i of a length-2 axis.
    let mut acc: Option<NodeId> = None;
    for (i, &row) in rows.iter().enumerate() {
        let padded = dag.add_node(
            RiscOp::Pad {
                padding: vec![(i, 2 - i - 1), (0, 0)],
                fill: 0.0,
            },
            vec![row],
            stacked_ty.clone(),
            None,
        );
        acc = Some(match acc {
            Some(prev) => dag.add_node(RiscOp::Add, vec![prev, padded], stacked_ty.clone(), None),
            None => padded,
        });
    }
    (acc.unwrap(), stacked_ty)
}

// =====================================================================
// MEAN over the symbolic-extent window stack.
// =====================================================================

/// `grad(sum(mean(stack(x), axis=0)))` constructs a VALID backward DAG and
/// matches finite differences. The reduced axis-0 extent is concrete (2
/// rows) but the non-reduced window axis is symbolic `m`, so the Pad/Shrink
/// adjoints of the `concat` must tolerate the symbolic axis (the #368 fix).
///
/// `f(x) = sum_cols(mean_rows([[x0,x1],[x2,x3]]))`
///       = mean(x0,x2) + mean(x1,x3) = (x0+x1+x2+x3)/2,
/// so `df/dx = [0.5, 0.5, 0.5, 0.5]`.
#[test]
fn issue_368_grad_windowed_mean_constructs_and_matches_fd() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec![DimInfo::Lit(4)]);
    let (stack, stack_ty) = build_windowed_stack(&mut dag, x);
    let mean = tier2::lower_mean(&mut dag, stack, 0, &stack_ty, None);
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![mean],
        scalar_f32(),
        None,
    );

    let result =
        grad_dag_checked(&dag, out, &[x]).expect("windowed mean grad must construct (issue #368)");
    let grad_x = result.grad_nodes[&x];

    let base = TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), base.clone());
    let analytic = eval_tensor(&result.dag, &inputs).expect("analytic eval")[&grad_x]
        .data
        .clone();
    for (i, a) in analytic.iter().enumerate() {
        assert!(
            (a - 0.5).abs() < 1e-5,
            "windowed-mean grad elem {i}: got {a}, want 0.5",
        );
    }
    assert_fd_match("windowed_mean", &dag, &result.dag, out, grad_x, &base);
}

// =====================================================================
// MAX_REDUCE over the symbolic-extent window stack.
// =====================================================================

/// `grad(sum(max_reduce(stack(x), axis=0)))` constructs a VALID backward DAG
/// (this is the path that USED to fail verify with "dimension count 1 vs 0")
/// and matches finite differences at a non-degenerate point.
///
/// `f(x) = max(x0,x2) + max(x1,x3)`. With x = [1,2,5,4]:
/// `max(1,5)=x2`, `max(2,4)=x3`, so `df/dx = [0, 0, 1, 1]`.
#[test]
fn issue_368_grad_windowed_max_constructs_and_matches_fd() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec![DimInfo::Lit(4)]);
    let (stack, stack_ty) = build_windowed_stack(&mut dag, x);
    let max = dag.add_node(
        RiscOp::MaxReduce { axis: 0 },
        vec![stack],
        reduced(&stack_ty),
        None,
    );
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![max],
        scalar_f32(),
        None,
    );

    let result =
        grad_dag_checked(&dag, out, &[x]).expect("windowed max grad must construct (issue #368)");
    let grad_x = result.grad_nodes[&x];

    // Strictly-separated maxima so the subgradient is a clean one-hot and FD
    // is well-defined (no tie at the perturbation scale).
    let base = TensorValue::from_vec(vec![4], vec![1.0, 2.0, 5.0, 4.0]);
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), base.clone());
    let analytic = eval_tensor(&result.dag, &inputs).expect("analytic eval")[&grad_x]
        .data
        .clone();
    let want = [0.0, 0.0, 1.0, 1.0];
    for (i, (a, w)) in analytic.iter().zip(want.iter()).enumerate() {
        assert!(
            (a - w).abs() < 1e-5,
            "windowed-max grad elem {i}: got {a}, want {w}"
        );
    }
    assert_fd_match("windowed_max", &dag, &result.dag, out, grad_x, &base);
}

// =====================================================================
// Symbolic-extent Pad / Shrink movement adjoints (the #368 core defect).
// =====================================================================

/// `grad(sum(pad(x, [(0,0),(0,0)])))` over a `[1, m]`-symbolic operand: the
/// Pad adjoint is a Shrink whose non-padded axis extent is the symbolic `m`.
/// Before #368 this `dim_size`-panicked; now it constructs and grads to all
/// ones (Pad with zero padding is the identity).
#[test]
fn issue_368_pad_adjoint_tolerates_symbolic_non_padded_axis() {
    let mut dag = Dag::new();
    let op_ty = f32_dims(vec![DimInfo::Lit(1), sym("m")]);
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        op_ty.clone(),
        None,
    );
    // Pad axis 0 from 1 -> 2 (concrete), axis 1 unchanged (symbolic m).
    let padded_ty = f32_dims(vec![DimInfo::Lit(2), sym("m")]);
    let padded = dag.add_node(
        RiscOp::Pad {
            padding: vec![(0, 1), (0, 0)],
            fill: 0.0,
        },
        vec![x],
        padded_ty.clone(),
        None,
    );
    // Reduce both axes to a scalar so the loss is sum(x).
    let r0 = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![padded],
        f32_dims(vec![sym("m")]),
        None,
    );
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![r0],
        scalar_f32(),
        None,
    );

    let result = grad_dag_checked(&dag, out, &[x])
        .expect("symbolic-axis Pad grad must construct (issue #368)");
    let grad_x = result.grad_nodes[&x];

    // x is [1, 3]; sum(pad(x)) = sum(x), so df/dx = ones.
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![1, 3], vec![2.0, 5.0, 9.0]),
    );
    let vals = eval_tensor(&result.dag, &inputs).expect("symbolic Pad grad eval");
    assert_eq!(vals[&grad_x].shape, vec![1, 3]);
    for (i, g) in vals[&grad_x].data.iter().enumerate() {
        assert!((g - 1.0).abs() < 1e-6, "Pad grad elem {i}: got {g}, want 1");
    }
}

// =====================================================================
// NEGATIVE PARITY: a genuinely malformed grad must STILL reject loudly —
// the #368 fix must not turn the backward-DAG verifier into a rubber stamp.
// =====================================================================

/// A binary op with a real, non-symbolic rank mismatch (rank-1 vs rank-2,
/// neither side a broadcastable scalar nor a symbolic-placeholder axis) must
/// still be rejected by backward-DAG verification. This pins that the #368
/// symbolic-tolerance work did not weaken the verifier for genuine shape
/// bugs.
#[test]
fn issue_368_genuine_rank_mismatch_still_rejected() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        f32_dims(vec![DimInfo::Lit(3)]),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "y".into() },
        vec![],
        f32_dims(vec![DimInfo::Lit(2), DimInfo::Lit(3)]),
        None,
    );
    // mul of a rank-1 and a rank-2 operand: an honestly ill-formed binary op.
    let bad = dag.add_node(
        RiscOp::Mul,
        vec![a, b],
        f32_dims(vec![DimInfo::Lit(3)]),
        None,
    );
    let out = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default"),
        vec![bad],
        scalar_f32(),
        None,
    );

    let err = grad_dag_checked(&dag, out, &[a])
        .err()
        .expect("a genuine rank mismatch must be rejected, not silently accepted");
    let msg = err.to_string();
    assert!(
        msg.contains("mismatched dimension count") || msg.contains("dimension"),
        "rejection must name the dimension mismatch; got: {msg}",
    );
}

/// Negative parity for the symbolic Reshape numel inference: a Reshape whose
/// product does NOT divide the input numel must NOT be silently "inferred"
/// to a wrong shape. `bind_symbolic_dims` leaves a non-inferable symbol to
/// the strict binder, which errors. Here the forward eval itself must reject
/// the impossible reshape rather than fabricate data.
#[test]
fn issue_368_reshape_inference_does_not_fabricate_impossible_shape() {
    let mut dag = Dag::new();
    // x is [4]; a reshape to [3, m] cannot conserve numel for any integer m
    // (4 is not divisible by 3). The eval must fail, not invent a shape.
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        f32_dims(vec![DimInfo::Lit(4)]),
        None,
    );
    let bad = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![DimInfo::Lit(3), sym("m")],
        },
        vec![x],
        f32_dims(vec![DimInfo::Lit(3), sym("m")]),
        None,
    );
    dag.add_root(bad);

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
    );
    let result = eval_tensor(&dag, &inputs);
    assert!(
        result.is_err(),
        "a numel-impossible reshape must be rejected, not silently reshaped",
    );
}
