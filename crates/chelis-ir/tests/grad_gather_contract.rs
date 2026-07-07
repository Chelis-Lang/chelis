//! Contract test for AD through the §3.5 gather→one-hot lowering.
//!
//! Spec/05-risc-primitives.md §3.5 says `gather(x, idx, axis)` decomposes via
//! `reshape + expand + mul + sum`. The framework's promise that "AD is correct
//! by construction" depends on that decomposition being differentiable through
//! the existing RISC adjoints — duplicate indices in the index tensor must
//! produce duplicate forward contributions which the backward pass through
//! `Sum`→`Expand` automatically scatter-adds. No hand-written backward.
//!
//! This test builds the post-§3.5 RISC DAG by hand for a 3-token / 2-vocab
//! gather where every token routes to vocab=0 (the MoE / embedding stress
//! case), and verifies that `grad_dag_checked` accumulates the duplicate-row
//! gradients correctly. It is the durable defense against the "library author
//! wrote a naive backward and silently dropped 99/100 of the batch" bug class
//! that motivated the linearity feature in the first place.
//!
//! The hand-built §3.5 decomposition test is independent of whether Surf-level
//! `gather` reaches IR directly. This branch also adds a first-class
//! `RiscOp::Gather` / `RiscOp::ScatterAdd` slice and verifies that its adjoint
//! preserves duplicate-index accumulation. Surf `gather` is still host-lane
//! unless a future change wires the dense lowering and sparse recognizer
//! together; that future path must keep these duplicate-index assertions green.
//!
//! Implementation note on `Expand`: Chelis's `RiscOp::Expand` adjoint
//! (`grad.rs`) wires only the rank-increasing form (insert a new axis), not
//! the in-place size-1 → larger form. The §3.5 lowering as written in spec
//! reads as size-1 broadcast, but the rank-increasing form is equivalent for
//! AD purposes and is what's currently differentiable, so this test uses it.

use std::collections::HashMap;

use chelis_ir::dag::{Bound, Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_with};
use chelis_ir::grad::grad_dag_checked;
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

#[test]
fn gather_via_section_3_5_lowering_accumulates_duplicate_indices() {
    let mut dag = Dag::new();

    // table: [vocab=2, dim=2] — the input we differentiate against.
    let table = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        t(vec![2, 2]),
        None,
    );

    // one_hot encoding of indices=[0, 0, 0]: build via Const+Pad.
    //   start: [n=3, 1] of 1.0
    let oh_col = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], t(vec![3, 1]), None);
    //   pad axis=1 by (0, 1) with fill 0 → [n=3, vocab=2] = [[1,0],[1,0],[1,0]]
    let one_hot = dag.add_node(
        RiscOp::Pad {
            padding: vec![
                (Bound::Lit(0), Bound::Lit(0)),
                (Bound::Lit(0), Bound::Lit(1)),
            ],
            fill: 0.0,
        },
        vec![oh_col],
        t(vec![3, 2]),
        None,
    );

    // Insert new axis at position 2 with size dim=2 → [n=3, vocab=2, dim=2].
    let oh_exp = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(2),
        },
        vec![one_hot],
        t(vec![3, 2, 2]),
        None,
    );

    // Insert new axis at position 0 with size n=3 → [n=3, vocab=2, dim=2].
    let table_exp = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(3),
        },
        vec![table],
        t(vec![3, 2, 2]),
        None,
    );

    // Mul + sum-over-vocab → [n=3, dim=2] (this IS the gather output).
    let product = dag.add_node(RiscOp::Mul, vec![oh_exp, table_exp], t(vec![3, 2, 2]), None);
    let gathered = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![product],
        t(vec![3, 2]),
        None,
    );

    // Collapse to scalar via two sum reductions to drive a scalar output.
    let s1 = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![gathered],
        t(vec![2]),
        None,
    );
    let s2 = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![s1],
        TensorType::scalar_f32(),
        None,
    );

    // Sanity: forward dag should be structurally valid.
    let fwd_errs = chelis_ir::verify::verify(&dag);
    assert!(
        fwd_errs.is_empty(),
        "forward dag verification errors: {fwd_errs:?}"
    );

    // Forward sanity check: sum over the gather of [[1,0],[1,0],[1,0]] @ table
    // is 3*(table[0,0] + table[0,1]) = 3*(1+2) = 9.
    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert(
        "table".to_string(),
        TensorValue {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
        },
    );
    let fwd_vals = eval_tensor_with(&dag, |n| inputs.get(n).cloned()).expect("forward eval");
    let fwd_out = &fwd_vals[&s2];
    assert!(
        (fwd_out.data[0] - 9.0).abs() < 1e-6,
        "forward sanity: expected 3*(1+2)=9.0, got {}",
        fwd_out.data[0]
    );

    // Backward: differentiate the scalar w.r.t. table.
    let grad = grad_dag_checked(&dag, s2, &[table]).expect("grad must succeed");
    let grad_node = grad.grad_nodes[&table];
    let bwd_vals = eval_tensor_with(&grad.dag, |n| inputs.get(n).cloned()).expect("backward eval");
    let dtable = &bwd_vals[&grad_node];

    // Expected: every token contributed +1 to dout/dtable[0,*]. Three tokens
    // routed to vocab=0, so the gradient at table[0,*] is 3 and at table[1,*]
    // is 0. If a future commit introduces a hand-rolled Gather adjoint that
    // forgets the duplicate-index accumulation, this assertion will reject it.
    assert_eq!(dtable.shape, vec![2, 2]);
    let expected = [3.0, 3.0, 0.0, 0.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (dtable.data[i] - want).abs() < 1e-6,
            "duplicate-index grad table[{i}]: expected {want}, got {} \
             (silent-drop bug? see crate-level docs)",
            dtable.data[i]
        );
    }
}

#[test]
fn first_class_gather_adjoint_scatter_add_accumulates_duplicate_indices() {
    let mut dag = Dag::new();
    let table = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        t(vec![2, 2]),
        None,
    );
    let indices = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], t_i32(vec![3]), None);
    let gathered = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![table, indices],
        t(vec![3, 2]),
        None,
    );
    let s1 = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![gathered],
        t(vec![2]),
        None,
    );
    let s2 = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![s1],
        TensorType::scalar_f32(),
        None,
    );

    let fwd_errs = chelis_ir::verify::verify(&dag);
    assert!(
        fwd_errs.is_empty(),
        "first-class gather verification errors: {fwd_errs:?}"
    );

    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert(
        "table".to_string(),
        TensorValue {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
        },
    );
    let grad = grad_dag_checked(&dag, s2, &[table]).expect("grad through gather must succeed");
    let grad_node = grad.grad_nodes[&table];
    let vals = eval_tensor_with(&grad.dag, |n| inputs.get(n).cloned()).expect("backward eval");
    let dtable = &vals[&grad_node];

    assert_eq!(dtable.shape, vec![2, 2]);
    let expected = [3.0, 3.0, 0.0, 0.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (dtable.data[i] - want).abs() < 1e-6,
            "first-class gather grad table[{i}]: expected {want}, got {}",
            dtable.data[i]
        );
    }
}

#[test]
fn first_class_gather_axis1_adjoint_scatter_add_accumulates_duplicate_indices() {
    let mut dag = Dag::new();
    let table = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        t(vec![2, 3]),
        None,
    );
    let indices = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], t_i32(vec![4]), None);
    let gathered = dag.add_node(
        RiscOp::Gather { axis: 1 },
        vec![table, indices],
        t(vec![2, 4]),
        None,
    );
    let s1 = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![gathered],
        t(vec![4]),
        None,
    );
    let s2 = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![s1],
        TensorType::scalar_f32(),
        None,
    );

    let fwd_errs = chelis_ir::verify::verify(&dag);
    assert!(
        fwd_errs.is_empty(),
        "axis-1 gather verification errors: {fwd_errs:?}"
    );

    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert(
        "table".to_string(),
        TensorValue {
            data: vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            shape: vec![2, 3],
        },
    );
    let grad = grad_dag_checked(&dag, s2, &[table]).expect("grad through axis-1 gather");
    let grad_node = grad.grad_nodes[&table];
    let vals = eval_tensor_with(&grad.dag, |n| inputs.get(n).cloned()).expect("backward eval");
    let dtable = &vals[&grad_node];

    assert_eq!(dtable.shape, vec![2, 3]);
    let expected = [4.0, 0.0, 0.0, 4.0, 0.0, 0.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (dtable.data[i] - want).abs() < 1e-6,
            "axis-1 gather grad table[{i}]: expected {want}, got {}",
            dtable.data[i]
        );
    }
}
