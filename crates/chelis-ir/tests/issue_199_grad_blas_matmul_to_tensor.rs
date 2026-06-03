//! Issue #199: `grad(loss, wrt=x)` cannot differentiate two common
//! constructs: a `BlasMatmul` node in the forward DAG, and any forward
//! that materializes a constant tensor literal via `to_tensor` in the
//! body of a differentiated function.
//!
//! Part 1 (this file): the `BlasMatmul` AD adjoint rule. The standard
//! matrix-multiply gradient is well-known:
//!
//!   For `Y = A @ B` with `A: [m, k]`, `B: [k, n]`, `Y: [m, n]`, and
//!   upstream gradient `dY: [m, n]`,
//!     `dA = dY @ B^T`  (shape `[m, k]`)
//!     `dB = A^T @ dY`  (shape `[k, n]`)
//!
//! Before the fix, `compute_adjoints` in `crates/chelis-ir/src/grad.rs`
//! returned `None` for `RiscOp::BlasMatmul { .. }`, which made any
//! forward DAG containing a `BlasMatmul` non-differentiable. The
//! gradient-construction failure surfaces as `grad_dag_checked` →
//! `AdError::NotSupported { op: "<unknown>", reason:
//! AdRejectionReason::Other("grad: failed to construct backward DAG
//! (unsupported op or verification failure)") }`.
//!
//! Part 2 (`to_tensor` in body) is deferred to Chelis-Lang/chelis#218.
//! The R1 -> R2 -> R3 red-team cascade against an initial in-tree
//! Part 2 attempt surfaced four distinct failure modes (reduction-
//! lowering wildcards, elementwise-lowering wildcards, list-element
//! Cons unification, and reshape Cons-chain shape extraction). Per
//! `feedback_rround_cascade_is_design_signal`, that pattern indicates
//! architectural redesign is needed rather than further patches.
//! #218 captures the full failure surface as design requirements for
//! a coordinated future implementation.

use chelis_ir::dag::{Dag, DimExpr, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::eval_tensor;
use chelis_ir::grad::{AdError, grad_dag, grad_dag_checked};
use chelis_types::types::Prim;
use std::collections::HashMap;

/// Build a forward DAG `Y = A @ B` (BlasMatmul) followed by a scalar
/// reduction `sum(sum(Y, 1), 0)`. Returns the DAG plus handles for the
/// load nodes (a, b) and the scalar output.
fn build_blas_matmul_scalar_loss(m: usize, k: usize, n: usize) -> (Dag, NodeId, NodeId, NodeId) {
    let mut dag = Dag::new();
    let a_ty = TensorType {
        dims: vec![DimInfo::Lit(m), DimInfo::Lit(k)],
        precision: Prim::F32,
    };
    let b_ty = TensorType {
        dims: vec![DimInfo::Lit(k), DimInfo::Lit(n)],
        precision: Prim::F32,
    };
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], a_ty, None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], b_ty, None);
    let c_ty = TensorType {
        dims: vec![DimInfo::Lit(m), DimInfo::Lit(n)],
        precision: Prim::F32,
    };
    let mm = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(m),
        DimExpr::Concrete(n),
        DimExpr::Concrete(k),
        Prim::F32,
    )
    .expect("matmul_default for f32 operands");
    let c = dag.add_node(mm, vec![a, b], c_ty, None);
    let row_sum_ty = TensorType {
        dims: vec![DimInfo::Lit(m)],
        precision: Prim::F32,
    };
    let s1 = dag.add_node(
        RiscOp::sum_default(1, Prim::F32).expect("sum_default for f32"),
        vec![c],
        row_sum_ty,
        None,
    );
    let scalar_ty = TensorType {
        dims: vec![],
        precision: Prim::F32,
    };
    let s2 = dag.add_node(
        RiscOp::sum_default(0, Prim::F32).expect("sum_default for f32"),
        vec![s1],
        scalar_ty,
        None,
    );
    (dag, a, b, s2)
}

/// Construct a column-major-as-vec row-major tensor value.
fn tensor_2d(data: Vec<Vec<f64>>) -> chelis_ir::eval::TensorValue {
    let rows = data.len();
    let cols = if rows == 0 { 0 } else { data[0].len() };
    let flat = data.into_iter().flatten().collect();
    chelis_ir::eval::TensorValue::from_vec(vec![rows, cols], flat)
}

fn tensor_eq_close(label: &str, got: &chelis_ir::eval::TensorValue, want: &[f64]) {
    assert_eq!(
        got.data.len(),
        want.len(),
        "{label}: shape mismatch: got {} elems, want {} elems",
        got.data.len(),
        want.len()
    );
    for (i, (g, w)) in got.data.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "{label}: elem {i} mismatch: got {g}, want {w}",
        );
    }
}

/// Positive: gradient w.r.t. A of `sum(A @ B)` should equal
/// `ones(m, n) @ B^T`, which is a row whose every column is the
/// column-sum of B repeated across rows. For B = [[1,0,0,0],
/// [0,1,0,0], [0,0,1,0]], `ones(2,4) @ B^T = ones(2,4) @ B^T` where
/// `B^T` has shape [4, 3]; `ones(2,4) @ B^T` has shape [2, 3]; every
/// element is `sum_n(1 * B[k,n]) = row_sum_of_B[k]`, which for the
/// chosen B is `[1, 1, 1]` for both rows.
#[test]
fn issue_199_blas_matmul_grad_wrt_lhs_is_finite_and_correct() {
    let (dag, a, _b, out) = build_blas_matmul_scalar_loss(2, 3, 4);
    let result = grad_dag_checked(&dag, out, &[a]).expect(
        "BlasMatmul gradient must succeed after the fix (closes Chelis-Lang/chelis#199 Part 1)",
    );
    let grad_a = result
        .grad_nodes
        .get(&a)
        .copied()
        .expect("gradient w.r.t. a must be present");

    // Set up inputs: a is [[1,2,3],[4,5,6]], b is selector matrix.
    let a_val = tensor_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let b_val = tensor_2d(vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ]);
    let mut inputs: HashMap<String, chelis_ir::eval::TensorValue> = HashMap::new();
    inputs.insert("a".into(), a_val);
    inputs.insert("b".into(), b_val);
    let values = eval_tensor(&result.dag, &inputs).expect("eval gradient DAG");
    let grad_a_val = values
        .get(&grad_a)
        .expect("grad-a node must evaluate to a tensor");
    // Expected: ones(2,4) @ B^T. B^T is shape [4,3] where the first
    // three rows are the standard basis e0,e1,e2 in R^3 and the last
    // row is all zeros. ones(2,4) @ B^T sums those rows once per
    // output row, giving [[1,1,1],[1,1,1]].
    tensor_eq_close("grad_a", grad_a_val, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn issue_199_blas_matmul_grad_wrt_rhs_is_finite_and_correct() {
    let (dag, _a, b, out) = build_blas_matmul_scalar_loss(2, 3, 4);
    let result = grad_dag_checked(&dag, out, &[b])
        .expect("BlasMatmul gradient must succeed for the rhs operand");
    let grad_b = result
        .grad_nodes
        .get(&b)
        .copied()
        .expect("gradient w.r.t. b must be present");
    let a_val = tensor_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let b_val = tensor_2d(vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ]);
    let mut inputs: HashMap<String, chelis_ir::eval::TensorValue> = HashMap::new();
    inputs.insert("a".into(), a_val);
    inputs.insert("b".into(), b_val);
    let values = eval_tensor(&result.dag, &inputs).expect("eval gradient DAG");
    let grad_b_val = values
        .get(&grad_b)
        .expect("grad-b node must evaluate to a tensor");
    // Expected: A^T @ ones(2,4). A^T is shape [3,2]; each row is the
    // column of A; multiplying by ones(2,4) gives each element of
    // A^T @ ones(2,4) at (k, n) = sum_m A[m,k] * 1 = col_sum(A, k).
    // col sums of A = [1+4, 2+5, 3+6] = [5, 7, 9]; result has shape
    // [3, 4] and every element of row k is col_sum(A, k):
    //   [[5,5,5,5],[7,7,7,7],[9,9,9,9]]
    tensor_eq_close(
        "grad_b",
        grad_b_val,
        &[5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0],
    );
}

/// Finite-difference check: per the agent contract
/// (feedback_evaluator_byte_identical_gate), the analytical gradient
/// must agree with a centered finite-difference approximation within
/// standard float tolerance.
#[test]
fn issue_199_blas_matmul_grad_matches_finite_difference() {
    let (dag, a, _b, out) = build_blas_matmul_scalar_loss(2, 3, 2);
    let result =
        grad_dag_checked(&dag, out, &[a]).expect("BlasMatmul gradient must succeed for wrt=a");
    let grad_a = result.grad_nodes[&a];
    let a_val = tensor_2d(vec![vec![0.7, -0.4, 1.1], vec![0.2, 0.9, -0.3]]);
    let b_val = tensor_2d(vec![vec![0.5, 1.2], vec![-0.7, 0.3], vec![0.1, -1.1]]);
    let mut inputs: HashMap<String, chelis_ir::eval::TensorValue> = HashMap::new();
    inputs.insert("a".into(), a_val.clone());
    inputs.insert("b".into(), b_val.clone());
    let analytical_values = eval_tensor(&result.dag, &inputs).expect("analytical eval");
    let analytical_grad = &analytical_values[&grad_a].data;

    // Numerical: central differences over each element of A.
    let h = 1e-3;
    let m = 2usize;
    let k = 3usize;
    let mut numerical = vec![0.0f64; m * k];
    for i in 0..m {
        for j in 0..k {
            let mut a_plus = a_val.clone();
            let mut a_minus = a_val.clone();
            let idx = i * k + j;
            a_plus.data[idx] += h;
            a_minus.data[idx] -= h;
            let mut inputs_p = inputs.clone();
            inputs_p.insert("a".into(), a_plus);
            let f_plus_map = eval_tensor(&dag, &inputs_p).expect("plus eval");
            let f_plus = f_plus_map[&out].data[0];
            let mut inputs_m = inputs.clone();
            inputs_m.insert("a".into(), a_minus);
            let f_minus_map = eval_tensor(&dag, &inputs_m).expect("minus eval");
            let f_minus = f_minus_map[&out].data[0];
            numerical[idx] = (f_plus - f_minus) / (2.0 * h);
        }
    }
    for (i, (an, num)) in analytical_grad.iter().zip(numerical.iter()).enumerate() {
        assert!(
            (an - num).abs() < 1e-3,
            "finite-diff mismatch at elem {i}: analytical {an}, numerical {num}",
        );
    }
}

/// Negative parity: gradient of a forward DAG whose root is a pure
/// constant (no Loads on the gradient path) must still produce a clean
/// result, not a panic or "non-differentiable" error. The adjoint of
/// every const is the empty input-gradient list, so a `grad(c, wrt=x)`
/// where `x` is unrelated should produce no grad_node entry for `x`,
/// not crash.
#[test]
fn issue_199_grad_with_unrelated_wrt_returns_no_grad_entry() {
    let mut dag = Dag::new();
    let scalar_ty = TensorType {
        dims: vec![],
        precision: Prim::F32,
    };
    let unrelated = dag.add_node(
        RiscOp::Load {
            name: "unrelated".into(),
        },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::F32,
        },
        None,
    );
    let c = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_ty, None);
    let result = grad_dag(&dag, c, &[unrelated]).expect(
        "grad of a pure constant w.r.t. an unrelated load should succeed; \
         the unrelated load simply receives no adjoint entry",
    );
    assert!(
        !result.grad_nodes.contains_key(&unrelated),
        "an unrelated wrt must not appear in grad_nodes; got {:?}",
        result.grad_nodes,
    );
}

/// Negative parity: `BlasMatmul` is no longer non-differentiable, so
/// `grad_dag_checked` must NOT emit the legacy "unsupported op or
/// verification failure" Other error for it. This pins the regression
/// surface: if a future refactor accidentally removes the BlasMatmul
/// adjoint arm and falls back through to the verifier-rejection path,
/// this test surfaces the regression with the legacy Other-error
/// shape.
#[test]
fn issue_199_blas_matmul_no_longer_emits_unknown_unsupported_error() {
    let (dag, a, _b, out) = build_blas_matmul_scalar_loss(2, 3, 4);
    let result = grad_dag_checked(&dag, out, &[a]);
    match result {
        Ok(_) => {}
        Err(AdError::NotSupported { op, reason }) => panic!(
            "BlasMatmul gradient unexpectedly rejected: op={op}, reason={reason:?}; \
             expected Ok after issue #199 Part 1 fix",
        ),
    }
}
