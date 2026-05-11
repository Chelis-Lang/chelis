//! Wave 5 red-team — Gap 3 / M4-residual AD edges through Gather, ScatterAdd,
//! and Scatter (replace).
//!
//! Existing locked tests in `grad_gather_contract.rs` and `scatter_replace_contract.rs`
//! cover:
//!   * All-duplicate indices through Gather on axis 0 + axis 1.
//!   * The Section §3.5 dense lowering's duplicate-index accumulation.
//!   * `Scatter` AD returns the structured `AdError::NotSupported` variant.
//!   * `Scatter` forward last-write-wins on duplicates and distinct indices.
//!   * Out-of-bounds axis is rejected by the verifier.
//!
//! This file adds adversarial coverage the existing suite doesn't have:
//!
//! 1. **All-distinct indices through Gather** on axis 0 → gradient is 1.0
//!    at each distinct row, NOT accumulation. Locks the non-duplicate path.
//! 2. **Mixed duplicates / distincts** through Gather on axis 0 → gradient
//!    exactly matches the per-index count. Verifies the adjoint sums in the
//!    expected positions.
//! 3. **Axis-1 mixed duplicates** through Gather on a wider table → exact
//!    per-column counts.
//! 4. **Out-of-range gather indices at FORWARD eval time** must panic with
//!    an "out of bounds" message, NOT silently wrap or zero. This locks the
//!    fail-closed evaluator contract from `eval.rs::gather`.
//! 5. **Out-of-range scatter_add indices at FORWARD eval time** must panic
//!    with "out of bounds". Locks the eval contract.
//! 6. **Out-of-range scatter (replace) indices at FORWARD eval time** must
//!    panic with "out of bounds". Locks the eval contract.
//! 7. **Scatter AD ignoring `wrt`**: `grad_dag_checked` returns the structured
//!    error regardless of which inputs are passed as `wrt`.
//! 8. **Display of structured AD error**: the rendered string for the
//!    `NonDeterministicAtDuplicateIndices` variant on `scatter_replace`
//!    includes the canonical user-facing language. Locks `Display` so
//!    downstream consumers that *do* render for humans (CLI diagnostics)
//!    don't silently regress.

use std::collections::HashMap;

use chelis_ir::DimInfo;
use chelis_ir::dag::{Dag, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_with};
use chelis_ir::grad::{AdError, AdRejectionReason, grad_dag_checked};
use chelis_types::types::Prim;

fn t(dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    }
}

fn t_i32(dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::Int32,
    }
}

fn build_gather_scalar(
    dag: &mut Dag,
    table_dims: Vec<usize>,
    indices_data: Vec<f64>,
    indices_shape: Vec<usize>,
    axis: usize,
    out_dims: Vec<usize>,
) -> (NodeId, NodeId) {
    let table = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        t(table_dims),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(indices_shape),
        None,
    );
    let gathered = dag.add_node(
        RiscOp::Gather { axis },
        vec![table, indices],
        t(out_dims.clone()),
        None,
    );
    // Reduce to scalar via repeated sums.
    let mut current = gathered;
    let mut current_shape = out_dims.clone();
    while !current_shape.is_empty() {
        let next_shape = current_shape[1..].to_vec();
        let ty = if next_shape.is_empty() {
            TensorType::scalar_f32()
        } else {
            t(next_shape.clone())
        };
        let next = dag.add_node(RiscOp::Sum { axis: 0 }, vec![current], ty, None);
        current = next;
        current_shape = next_shape;
    }
    let _ = indices_data; // not used here — callers pass via inputs map
    (table, current)
}

/// All-distinct indices through Gather on axis 0: gradient should be
/// exactly 1.0 at each gathered row, 0 elsewhere.
#[test]
fn gather_axis0_distinct_indices_gradient_is_one_per_picked_row() {
    let mut dag = Dag::new();
    let (table, out) = build_gather_scalar(
        &mut dag,
        vec![4, 2],          // table: [vocab=4, dim=2]
        vec![0.0, 2.0, 3.0], // distinct indices
        vec![3],
        0,
        vec![3, 2],
    );

    let grad = grad_dag_checked(&dag, out, &[table]).expect("grad");
    let grad_node = grad.grad_nodes[&table];

    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert(
        "table".to_string(),
        TensorValue {
            data: vec![1.0; 8],
            shape: vec![4, 2],
        },
    );
    inputs.insert(
        "indices".to_string(),
        TensorValue {
            data: vec![0.0, 2.0, 3.0],
            shape: vec![3],
        },
    );
    let vals = eval_tensor_with(&grad.dag, |n| inputs.get(n).cloned()).expect("bwd eval");
    let dtable = &vals[&grad_node];
    assert_eq!(dtable.shape, vec![4, 2]);
    // Row 0: picked once → 1; Row 1: never → 0; Row 2: picked once → 1;
    // Row 3: picked once → 1.
    let expected = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (dtable.data[i] - want).abs() < 1e-6,
            "distinct-index grad at table[{i}]: expected {want}, got {}",
            dtable.data[i]
        );
    }
}

/// Mixed duplicate / distinct indices: indices `[0, 0, 2, 0]` over a 4-row
/// table on axis 0 yields per-row counts `[3, 0, 1, 0]`. Gradient at
/// `table[i, *]` equals the count.
#[test]
fn gather_axis0_mixed_indices_gradient_matches_per_row_counts() {
    let mut dag = Dag::new();
    let (table, out) = build_gather_scalar(
        &mut dag,
        vec![4, 2],
        vec![0.0, 0.0, 2.0, 0.0],
        vec![4],
        0,
        vec![4, 2],
    );

    let grad = grad_dag_checked(&dag, out, &[table]).expect("grad");
    let grad_node = grad.grad_nodes[&table];

    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert(
        "table".to_string(),
        TensorValue {
            data: vec![1.0; 8],
            shape: vec![4, 2],
        },
    );
    inputs.insert(
        "indices".to_string(),
        TensorValue {
            data: vec![0.0, 0.0, 2.0, 0.0],
            shape: vec![4],
        },
    );
    let vals = eval_tensor_with(&grad.dag, |n| inputs.get(n).cloned()).expect("bwd eval");
    let dtable = &vals[&grad_node];
    assert_eq!(dtable.shape, vec![4, 2]);
    // Row counts: 0→3, 1→0, 2→1, 3→0 → grad per (row, dim) cell.
    let expected = [3.0, 3.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (dtable.data[i] - want).abs() < 1e-6,
            "mixed-index grad at table[{i}]: expected {want}, got {}",
            dtable.data[i]
        );
    }
}

/// Axis-1 gather: a wider table `[2, 5]` gathered with indices `[0, 2, 0, 4]`
/// on axis 1 yields per-column counts `[2, 0, 1, 0, 1]`. Each row's gradient
/// equals those counts.
#[test]
fn gather_axis1_mixed_indices_gradient_matches_per_column_counts() {
    let mut dag = Dag::new();
    let table = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        t(vec![2, 5]),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(vec![4]),
        None,
    );
    let gathered = dag.add_node(
        RiscOp::Gather { axis: 1 },
        vec![table, indices],
        t(vec![2, 4]),
        None,
    );
    let s1 = dag.add_node(RiscOp::Sum { axis: 0 }, vec![gathered], t(vec![4]), None);
    let s2 = dag.add_node(
        RiscOp::Sum { axis: 0 },
        vec![s1],
        TensorType::scalar_f32(),
        None,
    );

    let grad = grad_dag_checked(&dag, s2, &[table]).expect("grad");
    let grad_node = grad.grad_nodes[&table];

    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert(
        "table".to_string(),
        TensorValue {
            data: vec![1.0; 10],
            shape: vec![2, 5],
        },
    );
    inputs.insert(
        "indices".to_string(),
        TensorValue {
            data: vec![0.0, 2.0, 0.0, 4.0],
            shape: vec![4],
        },
    );
    let vals = eval_tensor_with(&grad.dag, |n| inputs.get(n).cloned()).expect("bwd eval");
    let dtable = &vals[&grad_node];
    assert_eq!(dtable.shape, vec![2, 5]);
    // Column counts: 0→2, 1→0, 2→1, 3→0, 4→1. Same per row.
    let expected = [2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (dtable.data[i] - want).abs() < 1e-6,
            "axis-1 mixed-index grad at table[{i}]: expected {want}, got {}",
            dtable.data[i]
        );
    }
}

/// Out-of-range gather index at FORWARD eval time must panic with
/// "out of bounds" — NOT silently wrap or produce zeros. Locks the
/// fail-closed eval contract at `eval.rs::gather:362`.
#[test]
#[should_panic(expected = "out of bounds")]
fn gather_eval_out_of_bounds_index_panics_fail_closed() {
    let mut dag = Dag::new();
    let table = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(vec![2]),
        None,
    );
    let _gathered = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![table, indices],
        t(vec![2, 2]),
        None,
    );

    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert(
        "table".to_string(),
        TensorValue {
            data: vec![1.0; 6],
            shape: vec![3, 2],
        },
    );
    // Index 5 is way out of range for a 3-row table.
    inputs.insert(
        "indices".to_string(),
        TensorValue {
            data: vec![1.0, 5.0],
            shape: vec![2],
        },
    );
    let _ = eval_tensor_with(&dag, |n| inputs.get(n).cloned()).unwrap();
}

/// Negative gather indices must also panic fail-closed.
#[test]
#[should_panic(expected = "out of bounds")]
fn gather_eval_negative_index_panics_fail_closed() {
    let mut dag = Dag::new();
    let table = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(vec![2]),
        None,
    );
    let _gathered = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![table, indices],
        t(vec![2, 2]),
        None,
    );

    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert(
        "table".to_string(),
        TensorValue {
            data: vec![1.0; 6],
            shape: vec![3, 2],
        },
    );
    inputs.insert(
        "indices".to_string(),
        TensorValue {
            data: vec![0.0, -1.0],
            shape: vec![2],
        },
    );
    let _ = eval_tensor_with(&dag, |n| inputs.get(n).cloned()).unwrap();
}

#[test]
#[should_panic(expected = "out of bounds")]
fn scatter_add_eval_out_of_bounds_index_panics_fail_closed() {
    let mut dag = Dag::new();
    let target = dag.add_node(
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(vec![2]),
        None,
    );
    let updates = dag.add_node(
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        t(vec![2, 2]),
        None,
    );
    let _sa = dag.add_node(
        RiscOp::ScatterAdd { axis: 0 },
        vec![target, indices, updates],
        t(vec![3, 2]),
        None,
    );

    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert(
        "target".to_string(),
        TensorValue {
            data: vec![0.0; 6],
            shape: vec![3, 2],
        },
    );
    inputs.insert(
        "indices".to_string(),
        TensorValue {
            data: vec![1.0, 9.0],
            shape: vec![2],
        },
    );
    inputs.insert(
        "updates".to_string(),
        TensorValue {
            data: vec![1.0; 4],
            shape: vec![2, 2],
        },
    );
    let _ = eval_tensor_with(&dag, |n| inputs.get(n).cloned()).unwrap();
}

#[test]
#[should_panic(expected = "out of bounds")]
fn scatter_replace_eval_out_of_bounds_index_panics_fail_closed() {
    let mut dag = Dag::new();
    let target = dag.add_node(
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(vec![2]),
        None,
    );
    let updates = dag.add_node(
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        t(vec![2, 2]),
        None,
    );
    let _sr = dag.add_node(
        RiscOp::Scatter { axis: 0 },
        vec![target, indices, updates],
        t(vec![3, 2]),
        None,
    );

    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert(
        "target".to_string(),
        TensorValue {
            data: vec![0.0; 6],
            shape: vec![3, 2],
        },
    );
    inputs.insert(
        "indices".to_string(),
        TensorValue {
            data: vec![1.0, 9.0],
            shape: vec![2],
        },
    );
    inputs.insert(
        "updates".to_string(),
        TensorValue {
            data: vec![1.0; 4],
            shape: vec![2, 2],
        },
    );
    let _ = eval_tensor_with(&dag, |n| inputs.get(n).cloned()).unwrap();
}

/// Scatter AD must reject regardless of `wrt`. The brief specifies the
/// pinned error shape; existing tests cover wrt=[target, updates]. Lock
/// the case where wrt=[target] only — the same structured rejection
/// must fire because the scatter is on the live forward graph.
#[test]
fn scatter_ad_rejects_regardless_of_wrt_subset() {
    let mut dag = Dag::new();
    let target = dag.add_node(
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], t_i32(vec![2]), None);
    let updates = dag.add_node(
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        t(vec![2, 2]),
        None,
    );
    let scatter = dag.add_node(
        RiscOp::Scatter { axis: 0 },
        vec![target, indices, updates],
        t(vec![3, 2]),
        None,
    );
    let s1 = dag.add_node(RiscOp::Sum { axis: 0 }, vec![scatter], t(vec![2]), None);
    let out = dag.add_node(
        RiscOp::Sum { axis: 0 },
        vec![s1],
        TensorType::scalar_f32(),
        None,
    );

    fn assert_rejects(result: Result<chelis_ir::grad::GradResult, AdError>, label: &str) {
        match result {
            Ok(_) => panic!("{label}: scatter AD must fail-closed but returned Ok(_)"),
            Err(err) => assert!(
                matches!(
                    err,
                    AdError::NotSupported {
                        op: "scatter_replace",
                        reason: AdRejectionReason::NonDeterministicAtDuplicateIndices,
                    }
                ),
                "{label}: expected pinned NotSupported{{scatter_replace,NonDeterministic...}}; got {err:?}"
            ),
        }
    }
    // wrt = [target] only — single-input subset.
    assert_rejects(grad_dag_checked(&dag, out, &[target]), "wrt=[target]");
    // wrt = [updates] only — same.
    assert_rejects(grad_dag_checked(&dag, out, &[updates]), "wrt=[updates]");
    // wrt = [] — empty subset. Must still detect Scatter on the live
    // forward graph and reject.
    assert_rejects(grad_dag_checked(&dag, out, &[]), "wrt=[]");
}

/// The Display rendering of the pinned AD-rejection variant must include
/// the canonical user-facing language so CLI diagnostics stay informative.
/// Locks the precise substrings — but NOT as the primary acceptance
/// (the primary acceptance is the variant match in `scatter_replace_contract.rs`).
#[test]
fn scatter_ad_error_display_contains_canonical_language() {
    let err = AdError::NotSupported {
        op: "scatter_replace",
        reason: AdRejectionReason::NonDeterministicAtDuplicateIndices,
    };
    let rendered = format!("{err}");
    // The Display impl in `grad.rs::AdError::fmt` mentions these exact phrases;
    // a regression that quietly changes the rendered text would be visible
    // to users reading diagnostics, so lock the durable phrases.
    let must_contain = [
        "scatter_replace",
        "non-deterministic",
        "duplicate",
        "scatter_add",
        "stop-gradient",
    ];
    for phrase in must_contain {
        assert!(
            rendered.contains(phrase),
            "AdError Display rendering missing canonical phrase `{phrase}` in:\n{rendered}"
        );
    }
}

/// Defensive: ScatterAdd's AD path remains None (the adjoint flows
/// through Gather elsewhere — see `grad.rs::backward_op`'s ScatterAdd
/// arm returns `None`). Locks the contract that ScatterAdd's own
/// backward is not auto-synthesized at this layer.
#[test]
fn scatter_add_backward_op_returns_no_individual_adjoint() {
    use chelis_ir::grad::grad_dag;
    let mut dag = Dag::new();
    let target = dag.add_node(
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], t_i32(vec![2]), None);
    let updates = dag.add_node(
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        t(vec![2, 2]),
        None,
    );
    let sa = dag.add_node(
        RiscOp::ScatterAdd { axis: 0 },
        vec![target, indices, updates],
        t(vec![3, 2]),
        None,
    );
    let s1 = dag.add_node(RiscOp::Sum { axis: 0 }, vec![sa], t(vec![2]), None);
    let out = dag.add_node(
        RiscOp::Sum { axis: 0 },
        vec![s1],
        TensorType::scalar_f32(),
        None,
    );

    // grad_dag(unchecked) returns None for ScatterAdd whose backward_op
    // path returns None (no individual adjoint synthesized at the op level).
    let result = grad_dag(&dag, out, &[updates]);
    assert!(
        result.is_none(),
        "grad_dag over ScatterAdd alone should return None at the op-adjoint layer; \
         got Some(_) (means the backward arm got rewired)"
    );
}
