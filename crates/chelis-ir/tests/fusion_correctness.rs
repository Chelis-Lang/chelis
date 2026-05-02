//! Fusion correctness tests (F1-F9).
//!
//! These verify the fusion pass produces correct DAGs and that fused evaluation
//! matches unfused evaluation via the IR evaluator.

use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_types::types::Prim;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn scalar_f32() -> TensorType {
    TensorType::scalar_f32()
}

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn mat_f32(rows: usize, cols: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision: Prim::F32,
    }
}

fn const_vec(dag: &mut Dag, value: f64, n: usize) -> NodeId {
    dag.add_node(RiscOp::Const { value }, vec![], vec_f32(n), None)
}

fn load(dag: &mut Dag, name: &str, ty: TensorType) -> NodeId {
    dag.add_node(RiscOp::Load { name: name.into() }, vec![], ty, None)
}

/// Evaluate a DAG with given inputs and return root outputs.
fn eval_dag(dag: &Dag, inputs: &HashMap<String, TensorValue>) -> Vec<TensorValue> {
    let roots: Vec<NodeId> = dag.roots().to_vec();
    let vals = eval_tensor_roots_with_strict(dag, &roots, |name| inputs.get(name).cloned())
        .expect("evaluation should succeed");
    // Return values for each root in order.
    roots.iter().map(|id| vals[id].clone()).collect()
}

/// Assert two TensorValue vecs are close within tolerance.
fn assert_outputs_close(a: &[TensorValue], b: &[TensorValue], tol: f64, context: &str) {
    assert_eq!(a.len(), b.len(), "{context}: different number of outputs");
    for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(va.shape, vb.shape, "{context} output {i}: shapes differ");
        for (j, (xa, xb)) in va.data.iter().zip(vb.data.iter()).enumerate() {
            assert!(
                (xa - xb).abs() < tol,
                "{context} output {i} element {j}: {xa} vs {xb} (diff {})",
                (xa - xb).abs()
            );
        }
    }
}

// ===========================================================================
// F1: add→neg fuses into single FusedElem
// ===========================================================================

#[test]
fn f1_add_neg_fuses() {
    let mut dag = Dag::new();
    let a = const_vec(&mut dag, 1.0, 4);
    let b = const_vec(&mut dag, 2.0, 4);
    let c = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4), None);
    let d = dag.add_node(RiscOp::Neg, vec![c], vec_f32(4), None);
    dag.add_root(d);

    let fused = chelis_ir::fuse::fuse(&dag);

    // The add→neg chain should be fused into fewer nodes than the original
    assert!(
        fused.len() < dag.len(),
        "Fused DAG should have fewer nodes: fused={} vs original={}",
        fused.len(),
        dag.len()
    );
    // Should contain a FusedElem node
    assert!(
        fused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::FusedElem { .. })),
        "Fused DAG should contain a FusedElem node"
    );
}

// ===========================================================================
// F2: add→relu→mul 3-way fuses
// ===========================================================================

#[test]
fn f2_three_way_chain_fuses() {
    let mut dag = Dag::new();
    let a = const_vec(&mut dag, 1.0, 4);
    let b = const_vec(&mut dag, 2.0, 4);
    let c = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4), None);
    // relu = max_elem(x, 0)
    let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], vec_f32(4), None);
    let relu = dag.add_node(RiscOp::MaxElem, vec![c, zero], vec_f32(4), None);
    let d = const_vec(&mut dag, 3.0, 4);
    let e = dag.add_node(RiscOp::Mul, vec![relu, d], vec_f32(4), None);
    dag.add_root(e);

    let fused = chelis_ir::fuse::fuse(&dag);

    // Should have at least one FusedElem
    let fused_count = fused
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, RiscOp::FusedElem { .. }))
        .count();
    assert!(fused_count >= 1, "Should have at least 1 FusedElem node");
}

// ===========================================================================
// F3: Fused output == unfused output (evaluator)
// ===========================================================================

#[test]
fn f3_fused_matches_unfused_evaluator() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    let c = const_vec(&mut dag, 2.0, 4);
    let added = dag.add_node(RiscOp::Add, vec![x, c], vec_f32(4), None);
    let negated = dag.add_node(RiscOp::Neg, vec![added], vec_f32(4), None);
    dag.add_root(negated);

    let fused = chelis_ir::fuse::fuse(&dag);

    let inputs: HashMap<String, TensorValue> = [(
        "x".to_string(),
        TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
    )]
    .into_iter()
    .collect();

    let unfused_out = eval_dag(&dag, &inputs);
    let fused_out = eval_dag(&fused, &inputs);

    assert_outputs_close(&unfused_out, &fused_out, 1e-6, "F3: add→neg");
}

// ===========================================================================
// F4: Multi-consumer NOT fused (would duplicate computation)
// ===========================================================================

#[test]
fn f4_multi_consumer_not_fused() {
    let mut dag = Dag::new();
    let a = const_vec(&mut dag, 1.0, 4);
    let b = const_vec(&mut dag, 2.0, 4);
    let shared = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4), None);
    // Two consumers of `shared`
    let left = dag.add_node(RiscOp::Neg, vec![shared], vec_f32(4), None);
    let right = dag.add_node(RiscOp::Exp, vec![shared], vec_f32(4), None);
    let out = dag.add_node(RiscOp::Add, vec![left, right], vec_f32(4), None);
    dag.add_root(out);

    let fused = chelis_ir::fuse::fuse(&dag);

    // `shared` has 2 consumers → must NOT be fused into either chain
    // Verify by evaluating: if fusion duplicated computation, output would still
    // be correct, but the invariant is violated. Check node count instead.
    let unfused_out = eval_dag(&dag, &HashMap::new());
    let fused_out = eval_dag(&fused, &HashMap::new());
    assert_outputs_close(&unfused_out, &fused_out, 1e-6, "F4: multi-consumer");

    // The `shared` add node should still exist as a materialized node (not absorbed)
    // because it has 2 consumers. There should be no FusedElem that contains 3+ steps
    // from the shared → left/right chains.
    for node in fused.nodes() {
        if let RiscOp::FusedElem { ref ops, .. } = node.op {
            assert!(
                ops.len() <= 2,
                "No FusedElem should span the multi-consumer boundary (got {} steps)",
                ops.len()
            );
        }
    }
}

// ===========================================================================
// F5: Elementwise→reduction fuses (add inlined into reduction inner loop)
// ===========================================================================

#[test]
fn f5_elementwise_into_reduction_fuses() {
    // Elementwise→reduction fusion is handled at emit time (option c):
    // the reduction kernel's inner loop inlines the preceding elementwise ops.
    // At the IR level, the elementwise chain still becomes a FusedElem node,
    // and the reduction consumes it. The DAG has fewer nodes because the
    // standalone Add is absorbed into a FusedElem (even if it's a 1-step chain
    // that the emitter can detect). But more importantly: correctness.
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", mat_f32(3, 4));
    let c = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4), None);
    let added = dag.add_node(RiscOp::Add, vec![x, c], mat_f32(3, 4), None);
    let summed = dag.add_node(RiscOp::Sum { axis: 1 }, vec![added], vec_f32(3), None);
    dag.add_root(summed);

    let fused = chelis_ir::fuse::fuse(&dag);

    // Verify correctness: fused matches unfused
    let inputs: HashMap<String, TensorValue> = [(
        "x".to_string(),
        TensorValue::from_vec(vec![3, 4], (0..12).map(|i| i as f64).collect()),
    )]
    .into_iter()
    .collect();

    let unfused_out = eval_dag(&dag, &inputs);
    let fused_out = eval_dag(&fused, &inputs);
    assert_outputs_close(&unfused_out, &fused_out, 1e-6, "F5: add→sum");
}

#[test]
fn f5_neg_reduction_into_elementwise_does_not_fuse() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", mat_f32(3, 4));
    let summed = dag.add_node(RiscOp::Sum { axis: 1 }, vec![x], vec_f32(3), None);
    let negated = dag.add_node(RiscOp::Neg, vec![summed], vec_f32(3), None);
    dag.add_root(negated);

    let fused = chelis_ir::fuse::fuse(&dag);

    // Reduction→elementwise should NOT fuse (iteration space changed)
    // The sum and neg should remain separate
    let inputs: HashMap<String, TensorValue> = [(
        "x".to_string(),
        TensorValue::from_vec(vec![3, 4], (0..12).map(|i| i as f64).collect()),
    )]
    .into_iter()
    .collect();

    let unfused_out = eval_dag(&dag, &inputs);
    let fused_out = eval_dag(&fused, &inputs);
    assert_outputs_close(&unfused_out, &fused_out, 1e-6, "F5 neg: sum→neg");
}

// ===========================================================================
// F6: Movement ops pass through (reshape in chain doesn't break fusion)
// ===========================================================================

#[test]
fn f6_movement_ops_pass_through() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(6));
    let reshaped = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
        },
        vec![x],
        mat_f32(2, 3),
        None,
    );
    let c = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3), None);
    let added = dag.add_node(RiscOp::Add, vec![reshaped, c], mat_f32(2, 3), None);
    let negated = dag.add_node(RiscOp::Neg, vec![added], mat_f32(2, 3), None);
    dag.add_root(negated);

    let fused = chelis_ir::fuse::fuse(&dag);

    // add→neg should still fuse (reshape is metadata-only, doesn't break chain)
    let inputs: HashMap<String, TensorValue> = [(
        "x".to_string(),
        TensorValue::from_vec(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )]
    .into_iter()
    .collect();

    let unfused_out = eval_dag(&dag, &inputs);
    let fused_out = eval_dag(&fused, &inputs);
    assert_outputs_close(&unfused_out, &fused_out, 1e-6, "F6: reshape + fusion");
}

// ===========================================================================
// F7: MNIST kernel count reduction
// ===========================================================================

#[test]
fn f7_mnist_fusion_reduces_nodes() {
    // Build a simplified MNIST-like DAG: matmul → add → relu → matmul → add
    // We can't test the full MNIST without the Surf parser, but we can build a
    // representative chain of ops.
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    let c1 = const_vec(&mut dag, 0.5, 4);
    let c2 = const_vec(&mut dag, 0.1, 4);
    // add → neg → exp → add (chain of 4 elementwise)
    let a = dag.add_node(RiscOp::Add, vec![x, c1], vec_f32(4), None);
    let b = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    let c = dag.add_node(RiscOp::Exp, vec![b], vec_f32(4), None);
    let d = dag.add_node(RiscOp::Add, vec![c, c2], vec_f32(4), None);
    dag.add_root(d);

    let original_len = dag.len();
    let fused = chelis_ir::fuse::fuse(&dag);

    assert!(
        fused.len() < original_len,
        "Fusion should reduce node count: {} → {}",
        original_len,
        fused.len()
    );
}

// ===========================================================================
// F8: Empty/trivial DAG unchanged
// ===========================================================================

#[test]
fn f8_trivial_dag_unchanged() {
    let mut dag = Dag::new();
    let c = dag.add_node(RiscOp::Const { value: 42.0 }, vec![], scalar_f32(), None);
    dag.add_root(c);

    let fused = chelis_ir::fuse::fuse(&dag);

    assert_eq!(
        fused.len(),
        dag.len(),
        "Trivial DAG should be unchanged by fusion"
    );
    // Should have no FusedElem nodes
    assert!(
        !fused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::FusedElem { .. })),
        "Trivial DAG should have no FusedElem"
    );
}

// ===========================================================================
// F10: Multi-consumer split — downstream chain fuses with multi-consumer as external input
// ===========================================================================

/// A (2 consumers) -> B -> C: B->C should fuse, A is an external input to the chain.
#[test]
fn f10_multi_consumer_downstream_chain_fuses() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    let y = load(&mut dag, "y", vec_f32(4));
    // A = add(x, y), has 2 consumers: B and D
    let a = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(4), None);
    // B = neg(A), single consumer: C
    let b = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    // C = exp(B), single consumer (root)
    let c = dag.add_node(RiscOp::Exp, vec![b], vec_f32(4), None);
    // D = sqrt(A), single consumer (root) — second consumer of A
    let d = dag.add_node(RiscOp::Sqrt, vec![a], vec_f32(4), None);
    dag.add_root(c);
    dag.add_root(d);

    let fused = chelis_ir::fuse::fuse(&dag);

    // B->C should fuse into a FusedElem (A is external input, not absorbed)
    let fused_elems: Vec<_> = fused
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, RiscOp::FusedElem { .. }))
        .collect();
    assert!(
        !fused_elems.is_empty(),
        "B->C should fuse into a FusedElem with A as external input"
    );

    // Verify correctness
    let inputs: HashMap<String, TensorValue> = [
        (
            "x".to_string(),
            TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
        ),
        (
            "y".to_string(),
            TensorValue::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]),
        ),
    ]
    .into_iter()
    .collect();

    let unfused_out = eval_dag(&dag, &inputs);
    let fused_out = eval_dag(&fused, &inputs);
    assert_outputs_close(&unfused_out, &fused_out, 1e-6, "F10: multi-consumer split");
}

// ===========================================================================
// F11: Both branches multi-consumer — nothing fuses
// ===========================================================================

/// A (2 consumers) -> B (2 consumers) -> C: No chain of length >= 2 possible.
#[test]
fn f11_double_multi_consumer_no_fusion() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    // A = neg(x), 2 consumers: B and E
    let a = dag.add_node(RiscOp::Neg, vec![x], vec_f32(4), None);
    // B = exp(A), 2 consumers: C and D
    let b = dag.add_node(RiscOp::Exp, vec![a], vec_f32(4), None);
    // C = sqrt(B)
    let c = dag.add_node(RiscOp::Sqrt, vec![b], vec_f32(4), None);
    // D = neg(B) — second consumer of B
    let d = dag.add_node(RiscOp::Neg, vec![b], vec_f32(4), None);
    // E = sin(A) — second consumer of A (use sin to avoid NaN from log of negatives)
    let e = dag.add_node(RiscOp::Sin, vec![a], vec_f32(4), None);
    dag.add_root(c);
    dag.add_root(d);
    dag.add_root(e);

    let fused = chelis_ir::fuse::fuse(&dag);

    // A has 2 consumers, B has 2 consumers — no node can be part of a 2+ chain
    // (each starts a chain of length 1 that gets discarded)
    let fused_elems: Vec<_> = fused
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, RiscOp::FusedElem { .. }))
        .collect();
    assert!(
        fused_elems.is_empty(),
        "No FusedElem should exist when all intermediate nodes have multiple consumers"
    );

    // Verify correctness
    let inputs: HashMap<String, TensorValue> = [(
        "x".to_string(),
        TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
    )]
    .into_iter()
    .collect();

    let unfused_out = eval_dag(&dag, &inputs);
    let fused_out = eval_dag(&fused, &inputs);
    assert_outputs_close(&unfused_out, &fused_out, 1e-6, "F11: double multi-consumer");
}

// ===========================================================================
// F12: Chain absorbs multi-consumer tail node (no duplication)
// ===========================================================================

/// A -> B (2 consumers) -> C and D: A->B fuses (B at chain tail), C and D
/// reference the FusedElem output which IS B's value.
#[test]
fn f12_multi_consumer_at_chain_tail() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    let c1 = const_vec(&mut dag, 2.0, 4);
    // A = add(x, c1), single consumer: B
    let a = dag.add_node(RiscOp::Add, vec![x, c1], vec_f32(4), None);
    // B = neg(A), 2 consumers: C and D
    let b = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    // C = exp(B)
    let c = dag.add_node(RiscOp::Exp, vec![b], vec_f32(4), None);
    // D = sin(B) — second consumer of B (use sin to avoid NaN from sqrt of negatives)
    let d = dag.add_node(RiscOp::Sin, vec![b], vec_f32(4), None);
    dag.add_root(c);
    dag.add_root(d);

    let fused = chelis_ir::fuse::fuse(&dag);

    // A->B should fuse: A has 1 consumer, B is the chain tail.
    // B has 2 consumers but that only prevents the chain from EXTENDING past B.
    // The chain [A, B] is valid — no computation duplication.
    let fused_elems: Vec<_> = fused
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, RiscOp::FusedElem { .. }))
        .collect();
    assert!(
        !fused_elems.is_empty(),
        "A->B should fuse even though B has multiple consumers (B is at chain tail)"
    );
    // The FusedElem should have exactly 2 steps (A, B)
    for fe in &fused_elems {
        if let RiscOp::FusedElem { ref ops } = fe.op {
            assert_eq!(ops.len(), 2, "Chain should be [A, B] = 2 steps");
        }
    }

    // Verify correctness — both outputs must match
    let inputs: HashMap<String, TensorValue> = [(
        "x".to_string(),
        TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
    )]
    .into_iter()
    .collect();

    let unfused_out = eval_dag(&dag, &inputs);
    let fused_out = eval_dag(&fused, &inputs);
    assert_outputs_close(&unfused_out, &fused_out, 1e-6, "F12: multi-consumer tail");
}

// ===========================================================================
// F13: Multi-consumer node available as external input to downstream chains
// ===========================================================================

/// A -> B (2 consumers) -> C -> D and B -> E. Verify C->D fuses with B as external input.
#[test]
fn f13_multi_consumer_as_external_input() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    // A = neg(x), single consumer: B
    let a = dag.add_node(RiscOp::Neg, vec![x], vec_f32(4), None);
    // B = exp(A), 2 consumers: C and E
    let b = dag.add_node(RiscOp::Exp, vec![a], vec_f32(4), None);
    // C = sqrt(B), single consumer: D
    let c = dag.add_node(RiscOp::Sqrt, vec![b], vec_f32(4), None);
    // D = neg(C), single consumer (root)
    let d = dag.add_node(RiscOp::Neg, vec![c], vec_f32(4), None);
    // E = log(B), single consumer (root) — second consumer of B
    let e = dag.add_node(RiscOp::Log, vec![b], vec_f32(4), None);
    dag.add_root(d);
    dag.add_root(e);

    let fused = chelis_ir::fuse::fuse(&dag);

    // A->B should fuse (A has 1 consumer). Chain stops at B (2 consumers).
    // C->D should fuse (C has 1 consumer, D has 1 consumer). B is external input.
    let fused_elems: Vec<_> = fused
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, RiscOp::FusedElem { .. }))
        .collect();
    assert!(
        fused_elems.len() >= 2,
        "Should have at least 2 FusedElem nodes: [A,B] and [C,D], got {}",
        fused_elems.len()
    );

    // Verify correctness
    let inputs: HashMap<String, TensorValue> = [(
        "x".to_string(),
        TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
    )]
    .into_iter()
    .collect();

    let unfused_out = eval_dag(&dag, &inputs);
    let fused_out = eval_dag(&fused, &inputs);
    assert_outputs_close(
        &unfused_out,
        &fused_out,
        1e-6,
        "F13: multi-consumer external input",
    );
}

// ===========================================================================
// F9: Fan-in: two inputs to one fused chain
// ===========================================================================

#[test]
fn f9_fan_in_two_inputs() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    let y = load(&mut dag, "y", vec_f32(4));
    let added = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(4), None);
    let negated = dag.add_node(RiscOp::Neg, vec![added], vec_f32(4), None);
    dag.add_root(negated);

    let fused = chelis_ir::fuse::fuse(&dag);

    let inputs: HashMap<String, TensorValue> = [
        (
            "x".to_string(),
            TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
        ),
        (
            "y".to_string(),
            TensorValue::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]),
        ),
    ]
    .into_iter()
    .collect();

    let unfused_out = eval_dag(&dag, &inputs);
    let fused_out = eval_dag(&fused, &inputs);
    assert_outputs_close(&unfused_out, &fused_out, 1e-6, "F9: fan-in add→neg");
}

// ===========================================================================
// FR1: reduction_inlined_fused_elems identifies single-consumer FusedElem→reduction
// ===========================================================================

#[test]
fn fr1_reduction_inlined_identifies_fused_elem_into_sum() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", mat_f32(3, 4));
    let c = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4), None);
    let added = dag.add_node(RiscOp::Add, vec![x, c], mat_f32(3, 4), None);
    let summed = dag.add_node(RiscOp::Sum { axis: 1 }, vec![added], vec_f32(3), None);
    dag.add_root(summed);

    let fused = chelis_ir::fuse::fuse(&dag);
    let inlined = chelis_ir::fuse::reduction_inlined_fused_elems(&fused);

    // The add was the only elementwise op, so it might or might not form a FusedElem
    // (chains need >= 2 nodes). But whatever the DAG structure, if a FusedElem feeds
    // the sum with no other consumers, it should be in the inlined set.
    // If the Add didn't fuse (single op), inlined should be empty (no FusedElem node).
    // Let's verify the fused DAG structure first.
    let has_fused_elem = fused
        .nodes()
        .iter()
        .any(|n| matches!(n.op, RiscOp::FusedElem { .. }));

    if has_fused_elem {
        assert!(
            !inlined.is_empty(),
            "Single-consumer FusedElem feeding Sum should be reduction-inlined"
        );
    }
    // Either way, correctness holds:
    let inputs: HashMap<String, TensorValue> = [(
        "x".to_string(),
        TensorValue::from_vec(vec![3, 4], (0..12).map(|i| i as f64).collect()),
    )]
    .into_iter()
    .collect();
    let unfused_out = eval_dag(&dag, &inputs);
    let fused_out = eval_dag(&fused, &inputs);
    assert_outputs_close(&unfused_out, &fused_out, 1e-6, "FR1: add→sum fused");
}

#[test]
fn fr2_multi_consumer_fused_elem_not_inlined() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", mat_f32(3, 4));
    let c = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4), None);
    let added = dag.add_node(RiscOp::Add, vec![x, c], mat_f32(3, 4), None);
    let negated = dag.add_node(RiscOp::Neg, vec![added], mat_f32(3, 4), None);
    // Two consumers of negated: root + sum
    let summed = dag.add_node(RiscOp::Sum { axis: 1 }, vec![negated], vec_f32(3), None);
    dag.add_root(negated);
    dag.add_root(summed);

    let fused = chelis_ir::fuse::fuse(&dag);
    let inlined = chelis_ir::fuse::reduction_inlined_fused_elems(&fused);

    // The FusedElem has 2 consumers (root + sum) → should NOT be inlined.
    assert!(
        inlined.is_empty(),
        "Multi-consumer FusedElem should NOT be reduction-inlined"
    );
}

#[test]
fn fr3_chain_into_sum_correctness() {
    // add→neg→sum: the add→neg chain fuses into FusedElem, which then feeds sum.
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", mat_f32(3, 4));
    let c = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4), None);
    let added = dag.add_node(RiscOp::Add, vec![x, c], mat_f32(3, 4), None);
    let negated = dag.add_node(RiscOp::Neg, vec![added], mat_f32(3, 4), None);
    let summed = dag.add_node(RiscOp::Sum { axis: 1 }, vec![negated], vec_f32(3), None);
    dag.add_root(summed);

    let fused = chelis_ir::fuse::fuse(&dag);
    let inlined = chelis_ir::fuse::reduction_inlined_fused_elems(&fused);

    // add→neg should fuse into FusedElem, which feeds sum with 1 consumer → inlined.
    assert!(
        !inlined.is_empty(),
        "FusedElem(add→neg) feeding Sum should be reduction-inlined"
    );

    let inputs: HashMap<String, TensorValue> = [(
        "x".to_string(),
        TensorValue::from_vec(vec![3, 4], (0..12).map(|i| i as f64).collect()),
    )]
    .into_iter()
    .collect();
    let unfused_out = eval_dag(&dag, &inputs);
    let fused_out = eval_dag(&fused, &inputs);
    assert_outputs_close(&unfused_out, &fused_out, 1e-6, "FR3: add→neg→sum");
}

#[test]
fn fr4_realize_is_fusion_barrier() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    let y = load(&mut dag, "y", vec_f32(4));
    let added = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(4), None);
    let realized = dag.add_node(RiscOp::Realize, vec![added], vec_f32(4), None);
    let negated = dag.add_node(RiscOp::Neg, vec![realized], vec_f32(4), None);
    dag.add_root(negated);

    let fused = chelis_ir::fuse::fuse(&dag);

    assert!(
        fused
            .nodes()
            .iter()
            .all(|node| !matches!(node.op, RiscOp::FusedElem { .. })),
        "realize() must block fusion across the materialization boundary"
    );

    let inputs: HashMap<String, TensorValue> = [
        (
            "x".to_string(),
            TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
        ),
        (
            "y".to_string(),
            TensorValue::from_vec(vec![4], vec![0.5, -1.0, 1.5, -2.0]),
        ),
    ]
    .into_iter()
    .collect();

    let unfused_out = eval_dag(&dag, &inputs);
    let fused_out = eval_dag(&fused, &inputs);
    assert_outputs_close(&unfused_out, &fused_out, 1e-6, "FR4: realize barrier");
}
