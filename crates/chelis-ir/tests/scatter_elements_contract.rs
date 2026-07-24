//! Contract test for the first-class element-wise replace-scatter
//! `RiscOp::ScatterElements { axis }` (ONNX `ScatterElements`,
//! `spec/05-risc-primitives.md` §3.5.1).
//!
//! Oracle, enforced by this file:
//!
//! 1. **Forward semantics.** Element-wise: `data`, `indices`, and
//!    `updates` share a rank; `indices.dims == updates.dims`;
//!    `output.dims == data.dims`. For each coordinate `c` over
//!    `indices`, `updates[c]` lands at `output[c with c[axis] :=
//!    indices[c]]`. Duplicate writes resolve last-write-wins under
//!    updates row-major flat order — distinct from the hyperplane
//!    `Scatter` (whose updates match
//!    `data.dims[..axis] ++ indices.dims ++ data.dims[axis+1..]`).
//!
//! 2. **AD policy is fail-closed with a structured error**, identical
//!    to `Scatter`: `AdError::NotSupported { op: "scatter_elements",
//!    reason: AdRejectionReason::NonDeterministicAtDuplicateIndices }`,
//!    asserted by enum pattern match (NOT `.contains()` on Display).
//!
//! 3. **Structural verification.** A well-formed graph verifies clean;
//!    a graph whose `indices.dims != updates.dims` is rejected (the
//!    element-wise contract, negative parity for part 1).

use std::collections::HashMap;

use chelis_ir::dag::{Dag, RiscOp};
use chelis_ir::eval::{TensorValue, eval_tensor_with};
use chelis_ir::grad::{AdError, AdRejectionReason, grad_dag_checked};
use chelis_ir::{DimInfo, TensorType};
use chelis_types::types::Prim;

// chelis#729 Phase 1: f64-typed fixtures. These tests pin the ONNX
// ScatterElements COORDINATE semantics with decimal probe values (1.1,
// 2.2); per-dtype finalize would round those at f32, which is not the
// subject here.
fn t(dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::F64,
    }
}

fn t_i32(dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::Int32,
    }
}

/// Part (1): forward semantics over a 2-D `data` with axis 0. This is
/// the canonical ONNX `ScatterElements` example shape: data [3,3],
/// indices/updates [2,3]. Each `(i, j)` writes `updates[i,j]` to
/// `output[indices[i,j], j]` — the column `j` is preserved from the
/// update coordinate, only the row is taken from the index.
#[test]
fn scatter_elements_forward_axis0_preserves_off_axis_coord() {
    let mut dag = Dag::new();
    let data = dag.add_node(
        RiscOp::Load {
            name: "data".into(),
        },
        vec![],
        t(vec![3, 3]),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(vec![2, 3]),
        None,
    );
    let updates = dag.add_node(
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        t(vec![2, 3]),
        None,
    );
    let scatter = dag.add_node(
        RiscOp::ScatterElements { axis: 0 },
        vec![data, indices, updates],
        t(vec![3, 3]),
        None,
    );

    let fwd_errs = chelis_ir::verify::verify(&dag);
    assert!(
        fwd_errs.is_empty(),
        "forward DAG verification errors: {fwd_errs:?}"
    );

    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert(
        "data".to_string(),
        TensorValue::from_vec(vec![3, 3], vec![0.0; 9]),
    );
    // indices (row-major over [2,3]):
    //   row0: [1, 0, 2]  row1: [0, 2, 1]
    inputs.insert(
        "indices".to_string(),
        TensorValue::from_vec(vec![2, 3], vec![1.0, 0.0, 2.0, 0.0, 2.0, 1.0]),
    );
    // updates (row-major over [2,3]):
    //   row0: [1.0, 1.1, 1.2]  row1: [2.0, 2.1, 2.2]
    inputs.insert(
        "updates".to_string(),
        TensorValue::from_vec(vec![2, 3], vec![1.0, 1.1, 1.2, 2.0, 2.1, 2.2]),
    );
    let vals = eval_tensor_with(&dag, |n| inputs.get(n).cloned()).expect("scatter_elements eval");
    let out = &vals[&scatter];

    // Hand-computed ONNX ScatterElements result:
    //   (0,0): idx 1, col 0 -> out[1,0]=1.0
    //   (0,1): idx 0, col 1 -> out[0,1]=1.1
    //   (0,2): idx 2, col 2 -> out[2,2]=1.2
    //   (1,0): idx 0, col 0 -> out[0,0]=2.0
    //   (1,1): idx 2, col 1 -> out[2,1]=2.1
    //   (1,2): idx 1, col 2 -> out[1,2]=2.2
    // output row-major [3,3]:
    //   [[2.0, 1.1, 0.0],
    //    [1.0, 0.0, 2.2],
    //    [0.0, 2.1, 1.2]]
    assert_eq!(out.shape, vec![3, 3]);
    let expected = [2.0, 1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 2.1, 1.2];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (out.to_f64_lossy_vec()[i] - want).abs() < 1e-9,
            "scatter_elements at flat index {i}: expected {want}, got {}",
            out.to_f64_lossy_vec()[i]
        );
    }
}

/// Part (1) — axis 1 variant: index substitutes the column, the row is
/// preserved from the update coordinate. Confirms the axis parameter is
/// honored (not hard-coded to 0).
#[test]
fn scatter_elements_forward_axis1_substitutes_column() {
    let mut dag = Dag::new();
    let data = dag.add_node(
        RiscOp::Load {
            name: "data".into(),
        },
        vec![],
        t(vec![2, 3]),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        t_i32(vec![2, 2]),
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
    let scatter = dag.add_node(
        RiscOp::ScatterElements { axis: 1 },
        vec![data, indices, updates],
        t(vec![2, 3]),
        None,
    );

    assert!(chelis_ir::verify::verify(&dag).is_empty());

    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert(
        "data".to_string(),
        TensorValue::from_vec(vec![2, 3], vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0]),
    );
    // indices [2,2]: row0 -> cols [2, 0]; row1 -> cols [1, 2]
    inputs.insert(
        "indices".to_string(),
        TensorValue::from_vec(vec![2, 2], vec![2.0, 0.0, 1.0, 2.0]),
    );
    inputs.insert(
        "updates".to_string(),
        TensorValue::from_vec(vec![2, 2], vec![100.0, 101.0, 200.0, 202.0]),
    );
    let vals = eval_tensor_with(&dag, |n| inputs.get(n).cloned()).expect("scatter_elements eval");
    let out = &vals[&scatter];

    // (0,0): col 2 -> out[0,2]=100.0
    // (0,1): col 0 -> out[0,0]=101.0
    // (1,0): col 1 -> out[1,1]=200.0
    // (1,1): col 2 -> out[1,2]=202.0
    // out row0: [101, 11, 100]; row1: [20, 200, 202]
    let expected = [101.0, 11.0, 100.0, 20.0, 200.0, 202.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (out.to_f64_lossy_vec()[i] - want).abs() < 1e-9,
            "scatter_elements axis1 flat {i}: expected {want}, got {}",
            out.to_f64_lossy_vec()[i]
        );
    }
}

/// Part (1) — duplicate-index last-write-wins under updates row-major
/// order. Two update positions target the same output cell; the one
/// with the larger flat index in `updates` must win.
#[test]
fn scatter_elements_forward_duplicate_last_write_wins() {
    let mut dag = Dag::new();
    let data = dag.add_node(
        RiscOp::Load {
            name: "data".into(),
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
        t_i32(vec![2, 2]),
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
    let scatter = dag.add_node(
        RiscOp::ScatterElements { axis: 0 },
        vec![data, indices, updates],
        t(vec![3, 2]),
        None,
    );

    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert(
        "data".to_string(),
        TensorValue::from_vec(vec![3, 2], vec![0.0; 6]),
    );
    // Both rows of indices point column 0 at row 1: positions (0,0) and
    // (1,0) collide on out[1,0]. Update (1,0) has the larger flat index
    // (2 > 0) and must win.
    inputs.insert(
        "indices".to_string(),
        TensorValue::from_vec(vec![2, 2], vec![1.0, 0.0, 1.0, 0.0]),
    );
    inputs.insert(
        "updates".to_string(),
        TensorValue::from_vec(vec![2, 2], vec![7.0, 8.0, 9.0, 10.0]),
    );
    let vals = eval_tensor_with(&dag, |n| inputs.get(n).cloned()).expect("scatter_elements eval");
    let out = &vals[&scatter];

    // out[1,0] receives update (0,0)=7.0 then (1,0)=9.0 -> 9.0 wins.
    // out[0,1] receives (0,1)=8.0; out[0,1] index is row 0, col 1.
    //   (0,1): idx 0 -> out[0,1]=8.0
    //   (1,1): idx 0 -> out[0,1]=10.0  (also collides! larger flat wins)
    // So out[0,1] = 10.0.
    // out: [[0, 10], [9, 0], [0, 0]]
    let expected = [0.0, 10.0, 9.0, 0.0, 0.0, 0.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (out.to_f64_lossy_vec()[i] - want).abs() < 1e-9,
            "scatter_elements duplicate flat {i}: expected {want}, got {}",
            out.to_f64_lossy_vec()[i]
        );
    }
}

/// Part (2): AD over `RiscOp::ScatterElements` must fail-closed with the
/// structured error, identical policy to `Scatter`.
#[test]
fn scatter_elements_ad_returns_structured_not_supported_error() {
    let mut dag = Dag::new();
    let data = dag.add_node(
        RiscOp::Load {
            name: "data".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Const { value: 0.0 },
        vec![],
        t_i32(vec![2, 2]),
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
    let scatter = dag.add_node(
        RiscOp::ScatterElements { axis: 0 },
        vec![data, indices, updates],
        t(vec![3, 2]),
        None,
    );
    let s1 = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![scatter],
        t(vec![2]),
        None,
    );
    let out = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![s1],
        TensorType::scalar_f32(),
        None,
    );

    let result = grad_dag_checked(&dag, out, &[data, updates]);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!(
            "AD over RiscOp::ScatterElements must fail-closed with \
             AdError::NotSupported; got Ok(_)"
        ),
    };
    assert!(
        matches!(
            err,
            AdError::NotSupported {
                op: "scatter_elements",
                reason: AdRejectionReason::NonDeterministicAtDuplicateIndices,
            }
        ),
        "AD must return exactly AdError::NotSupported {{ \
         op: \"scatter_elements\", reason: NonDeterministicAtDuplicateIndices \
         }}; got: {err:?}"
    );
}

/// Part (3) — negative parity: the element-wise contract requires
/// `indices.dims == updates.dims`. A graph that violates it (here the
/// hyperplane shape that `Scatter` would accept) must be rejected by
/// the verifier, naming `scatter_elements`.
#[test]
fn scatter_elements_verify_rejects_mismatched_index_update_dims() {
    let mut dag = Dag::new();
    let data = dag.add_node(
        RiscOp::Load {
            name: "data".into(),
        },
        vec![],
        t(vec![3, 2]),
        None,
    );
    // indices [2] (1-D), updates [2, 2] (2-D): the hyperplane shape that
    // Scatter accepts but ScatterElements must reject (different ranks
    // AND indices.dims != updates.dims).
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
    dag.add_node(
        RiscOp::ScatterElements { axis: 0 },
        vec![data, indices, updates],
        t(vec![3, 2]),
        None,
    );

    let errs = chelis_ir::verify::verify(&dag);
    assert!(
        errs.iter().any(|e| e.contains("scatter_elements")),
        "verifier must reject element-wise scatter with mismatched \
         indices/updates dims and name the op; got: {errs:?}"
    );
}
