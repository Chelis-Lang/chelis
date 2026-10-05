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
//! 4. **Out-of-range gather indices at FORWARD eval time** must fail with
//!    the [05-SPARSE-1] `Domain` trap, NOT silently wrap or zero, and never
//!    panic ([04-NUM-10]). This locks the fail-closed evaluator contract from
//!    `eval.rs::gather`.
//! 5. **Out-of-range scatter_add indices at FORWARD eval time** must fail
//!    with the same trap in `scatter`. Locks the eval contract.
//! 6. **Out-of-range scatter (replace) indices at FORWARD eval time** must
//!    fail with the same trap in `scatter_replace`. Locks the eval contract.
//! 7. **Scatter AD ignoring `wrt`**: `grad_dag_checked` returns the structured
//!    error regardless of which inputs are passed as `wrt`.
//! 8. **Display of structured AD error**: the rendered string for the
//!    `NonDeterministicAtDuplicateIndices` variant on `scatter_replace`
//!    includes the canonical user-facing language. Locks `Display` so
//!    downstream consumers that *do* render for humans (CLI diagnostics)
//!    don't silently regress.
//! 9. **ScatterAdd reverse mode**: target cotangents pass through, update
//!    cotangents gather at the same indices, and the discrete indices receive
//!    no cotangent.

use chelis_unord::UnordMap;

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
    decl: chelis_ir::dag::DeclId,
    table_dims: Vec<usize>,
    indices_data: Vec<f64>,
    indices_shape: Vec<usize>,
    axis: usize,
    out_dims: Vec<usize>,
) -> (NodeId, NodeId) {
    let table = dag.add_node(
        decl,
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        t(table_dims),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(indices_shape),
        None,
    );
    let gathered = dag.add_node(
        decl,
        RiscOp::Gather {
            axis,
            batch_rank: 0,
        },
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
        let next = dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 0,
                accumulator: Prim::F32,
            },
            vec![current],
            ty,
            None,
        );
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
    let decl = dag.declare("test");
    let (table, out) = build_gather_scalar(
        &mut dag,
        decl,
        vec![4, 2],          // table: [vocab=4, dim=2]
        vec![0.0, 2.0, 3.0], // distinct indices
        vec![3],
        0,
        vec![3, 2],
    );

    let grad = grad_dag_checked(&dag, out, &[table]).expect("grad");
    let grad_node = grad.grad_nodes[&table];

    let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
    inputs.insert(
        "table".to_string(),
        TensorValue::from_vec(vec![4, 2], vec![1.0; 8]),
    );
    inputs.insert(
        "indices".to_string(),
        TensorValue::from_vec(vec![3], vec![0.0, 2.0, 3.0]),
    );
    let vals = eval_tensor_with(&grad.dag, |n| inputs.get(n).cloned()).expect("bwd eval");
    let dtable = &vals[&grad_node];
    assert_eq!(dtable.shape, vec![4, 2]);
    // Row 0: picked once → 1; Row 1: never → 0; Row 2: picked once → 1;
    // Row 3: picked once → 1.
    let expected = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (dtable.to_f64_lossy_vec()[i] - want).abs() < 1e-6,
            "distinct-index grad at table[{i}]: expected {want}, got {}",
            dtable.to_f64_lossy_vec()[i]
        );
    }
}

/// Mixed duplicate / distinct indices: indices `[0, 0, 2, 0]` over a 4-row
/// table on axis 0 yields per-row counts `[3, 0, 1, 0]`. Gradient at
/// `table[i, *]` equals the count.
#[test]
fn gather_axis0_mixed_indices_gradient_matches_per_row_counts() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let (table, out) = build_gather_scalar(
        &mut dag,
        decl,
        vec![4, 2],
        vec![0.0, 0.0, 2.0, 0.0],
        vec![4],
        0,
        vec![4, 2],
    );

    let grad = grad_dag_checked(&dag, out, &[table]).expect("grad");
    let grad_node = grad.grad_nodes[&table];

    let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
    inputs.insert(
        "table".to_string(),
        TensorValue::from_vec(vec![4, 2], vec![1.0; 8]),
    );
    inputs.insert(
        "indices".to_string(),
        TensorValue::from_vec(vec![4], vec![0.0, 0.0, 2.0, 0.0]),
    );
    let vals = eval_tensor_with(&grad.dag, |n| inputs.get(n).cloned()).expect("bwd eval");
    let dtable = &vals[&grad_node];
    assert_eq!(dtable.shape, vec![4, 2]);
    // Row counts: 0→3, 1→0, 2→1, 3→0 → grad per (row, dim) cell.
    let expected = [3.0, 3.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (dtable.to_f64_lossy_vec()[i] - want).abs() < 1e-6,
            "mixed-index grad at table[{i}]: expected {want}, got {}",
            dtable.to_f64_lossy_vec()[i]
        );
    }
}

/// Axis-1 gather: a wider table `[2, 5]` gathered with indices `[0, 2, 0, 4]`
/// on axis 1 yields per-column counts `[2, 0, 1, 0, 1]`. Each row's gradient
/// equals those counts.
#[test]
fn gather_axis1_mixed_indices_gradient_matches_per_column_counts() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let table = dag.add_node(
        decl,
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        t(vec![2, 5]),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(vec![4]),
        None,
    );
    let gathered = dag.add_node(
        decl,
        RiscOp::Gather {
            axis: 1,
            batch_rank: 0,
        },
        vec![table, indices],
        t(vec![2, 4]),
        None,
    );
    let s1 = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![gathered],
        t(vec![4]),
        None,
    );
    let s2 = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![s1],
        TensorType::scalar_f32(),
        None,
    );

    let grad = grad_dag_checked(&dag, s2, &[table]).expect("grad");
    let grad_node = grad.grad_nodes[&table];

    let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
    inputs.insert(
        "table".to_string(),
        TensorValue::from_vec(vec![2, 5], vec![1.0; 10]),
    );
    inputs.insert(
        "indices".to_string(),
        TensorValue::from_vec(vec![4], vec![0.0, 2.0, 0.0, 4.0]),
    );
    let vals = eval_tensor_with(&grad.dag, |n| inputs.get(n).cloned()).expect("bwd eval");
    let dtable = &vals[&grad_node];
    assert_eq!(dtable.shape, vec![2, 5]);
    // Column counts: 0→2, 1→0, 2→1, 3→0, 4→1. Same per row.
    let expected = [2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (dtable.to_f64_lossy_vec()[i] - want).abs() < 1e-6,
            "axis-1 mixed-index grad at table[{i}]: expected {want}, got {}",
            dtable.to_f64_lossy_vec()[i]
        );
    }
}

/// Out-of-range gather index at FORWARD eval time must fail with the
/// sparse-index `Domain` trap — NOT silently wrap or produce zeros. Locks the
/// fail-closed eval contract at `eval.rs::gather`.
#[test]
fn gather_eval_out_of_bounds_index_traps_fail_closed() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let table = dag.add_node(
        decl,
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(vec![2]),
        None,
    );
    let _gathered = dag.add_node(
        decl,
        RiscOp::Gather {
            axis: 0,
            batch_rank: 0,
        },
        vec![table, indices],
        t(vec![2, 2]),
        None,
    );

    let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
    inputs.insert(
        "table".to_string(),
        TensorValue::from_vec(vec![3, 2], vec![1.0; 6]),
    );
    // Index 5 is way out of range for a 3-row table.
    inputs.insert(
        "indices".to_string(),
        TensorValue::from_vec(vec![2], vec![1.0, 5.0]),
    );
    let err = eval_tensor_with(&dag, |n| inputs.get(n).cloned())
        .expect_err("an index outside the axis must fail, never wrap or zero");
    assert!(
        err.contains("out of bounds") && err.ends_with("numeric trap: domain in gather at i64"),
        "{err}"
    );
}

/// Negative gather indices must also trap fail-closed.
#[test]
fn gather_eval_negative_index_traps_fail_closed() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let table = dag.add_node(
        decl,
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(vec![2]),
        None,
    );
    let _gathered = dag.add_node(
        decl,
        RiscOp::Gather {
            axis: 0,
            batch_rank: 0,
        },
        vec![table, indices],
        t(vec![2, 2]),
        None,
    );

    let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
    inputs.insert(
        "table".to_string(),
        TensorValue::from_vec(vec![3, 2], vec![1.0; 6]),
    );
    inputs.insert(
        "indices".to_string(),
        TensorValue::from_vec(vec![2], vec![0.0, -1.0]),
    );
    let err = eval_tensor_with(&dag, |n| inputs.get(n).cloned())
        .expect_err("an index outside the axis must fail, never wrap or zero");
    assert!(
        err.contains("out of bounds") && err.ends_with("numeric trap: domain in gather at i64"),
        "{err}"
    );
}

#[test]
fn scatter_add_eval_out_of_bounds_index_traps_fail_closed() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let target = dag.add_node(
        decl,
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(vec![2]),
        None,
    );
    let updates = dag.add_node(
        decl,
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        t(vec![2, 2]),
        None,
    );
    let _sa = dag.add_node(
        decl,
        RiscOp::ScatterAdd {
            axis: 0,
            batch_rank: 0,
        },
        vec![target, indices, updates],
        t(vec![3, 2]),
        None,
    );

    let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
    inputs.insert(
        "target".to_string(),
        TensorValue::from_vec(vec![3, 2], vec![0.0; 6]),
    );
    inputs.insert(
        "indices".to_string(),
        TensorValue::from_vec(vec![2], vec![1.0, 9.0]),
    );
    inputs.insert(
        "updates".to_string(),
        TensorValue::from_vec(vec![2, 2], vec![1.0; 4]),
    );
    let err = eval_tensor_with(&dag, |n| inputs.get(n).cloned())
        .expect_err("an index outside the axis must fail, never wrap or zero");
    assert!(
        err.contains("out of bounds") && err.ends_with("numeric trap: domain in scatter at i64"),
        "{err}"
    );
}

#[test]
fn scatter_replace_eval_out_of_bounds_index_traps_fail_closed() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let target = dag.add_node(
        decl,
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(vec![2]),
        None,
    );
    let updates = dag.add_node(
        decl,
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        t(vec![2, 2]),
        None,
    );
    let _sr = dag.add_node(
        decl,
        RiscOp::Scatter {
            axis: 0,
            batch_rank: 0,
        },
        vec![target, indices, updates],
        t(vec![3, 2]),
        None,
    );

    let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
    inputs.insert(
        "target".to_string(),
        TensorValue::from_vec(vec![3, 2], vec![0.0; 6]),
    );
    inputs.insert(
        "indices".to_string(),
        TensorValue::from_vec(vec![2], vec![1.0, 9.0]),
    );
    inputs.insert(
        "updates".to_string(),
        TensorValue::from_vec(vec![2, 2], vec![1.0; 4]),
    );
    let err = eval_tensor_with(&dag, |n| inputs.get(n).cloned())
        .expect_err("an index outside the axis must fail, never wrap or zero");
    assert!(
        err.contains("out of bounds")
            && err.ends_with("numeric trap: domain in scatter_replace at i64"),
        "{err}"
    );
}

/// Scatter AD must reject regardless of `wrt`. The brief specifies the
/// pinned error shape; existing tests cover wrt=[target, updates]. Lock
/// the case where wrt=[target] only — the same structured rejection
/// must fire because the scatter is on the live forward graph.
#[test]
fn scatter_ad_rejects_regardless_of_wrt_subset() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let target = dag.add_node(
        decl,
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::synth_const(t_i32(vec![2]).precision, 0.0),
        vec![],
        t_i32(vec![2]),
        None,
    );
    let updates = dag.add_node(
        decl,
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        t(vec![2, 2]),
        None,
    );
    let scatter = dag.add_node(
        decl,
        RiscOp::Scatter {
            axis: 0,
            batch_rank: 0,
        },
        vec![target, indices, updates],
        t(vec![3, 2]),
        None,
    );
    let s1 = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![scatter],
        t(vec![2]),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
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

/// ScatterAdd is linear in its target and updates: the target cotangent is
/// the upstream value and the updates cotangent gathers that value at the
/// forward indices. Its integer indices are discrete and receive no
/// cotangent even when explicitly requested.
#[test]
fn scatter_add_backward_routes_target_and_updates_but_not_indices() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let target = dag.add_node(
        decl,
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        t(vec![4]),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(vec![2]),
        None,
    );
    let updates = dag.add_node(
        decl,
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        t(vec![2]),
        None,
    );
    let sa = dag.add_node(
        decl,
        RiscOp::ScatterAdd {
            axis: 0,
            batch_rank: 0,
        },
        vec![target, indices, updates],
        t(vec![4]),
        None,
    );
    let coefficients = dag.add_node(
        decl,
        RiscOp::synth_const_tensor(Prim::F32, vec![2.0, 3.0, 5.0, 7.0]),
        vec![],
        t(vec![4]),
        None,
    );
    let weighted = dag.add_node(decl, RiscOp::Mul, vec![sa, coefficients], t(vec![4]), None);
    let out = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![weighted],
        TensorType::scalar_f32(),
        None,
    );

    let grad = grad_dag_checked(&dag, out, &[target, indices, updates])
        .expect("ScatterAdd reverse mode must be defined");
    assert!(grad.grad_nodes.contains_key(&target));
    assert!(grad.grad_nodes.contains_key(&updates));
    assert!(
        !grad.grad_nodes.contains_key(&indices),
        "integer ScatterAdd indices must remain a stop-gradient boundary"
    );

    let inputs = UnordMap::from([
        (
            "target".to_string(),
            TensorValue::from_vec(vec![4], vec![11.0, 13.0, 17.0, 19.0]),
        ),
        (
            "indices".to_string(),
            TensorValue::from_vec(vec![2], vec![1.0, 3.0]),
        ),
        (
            "updates".to_string(),
            TensorValue::from_vec(vec![2], vec![23.0, 29.0]),
        ),
    ]);
    let values = eval_tensor_with(&grad.dag, |name| inputs.get(name).cloned())
        .expect("ScatterAdd adjoints evaluate");
    assert_eq!(
        values[&grad.grad_nodes[&target]].to_f64_lossy_vec(),
        vec![2.0, 3.0, 5.0, 7.0]
    );
    assert_eq!(
        values[&grad.grad_nodes[&updates]].to_f64_lossy_vec(),
        vec![3.0, 7.0]
    );
}
