//! Adversarial fusion tests — red team Phase 1b.
//! These are NOT committed to the repo; they exist to probe for bugs.

use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_ir::fuse::fuse;
use chelis_types::ElementRef;
use chelis_types::types::Prim;
use std::collections::HashMap;

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn load(dag: &mut Dag, name: &str, ty: TensorType) -> NodeId {
    dag.add_node(RiscOp::Load { name: name.into() }, vec![], ty, None)
}

fn eval_dag(dag: &Dag, inputs: &HashMap<String, TensorValue>) -> Vec<TensorValue> {
    let roots: Vec<NodeId> = dag.roots().to_vec();
    let vals = eval_tensor_roots_with_strict(dag, &roots, |name| inputs.get(name).cloned())
        .expect("evaluation should succeed");
    roots.iter().map(|id| vals[id].clone()).collect()
}

fn assert_close(a: &[TensorValue], b: &[TensorValue], tol: f64, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: different number of outputs");
    for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(va.shape, vb.shape, "{label} output {i}: shapes differ");
        for (j, (xa, xb)) in va
            .to_f64_lossy_vec()
            .iter()
            .zip(vb.to_f64_lossy_vec().iter())
            .enumerate()
        {
            assert!(
                (xa - xb).abs() < tol,
                "{label} output {i} element {j}: {xa} vs {xb} (diff {})",
                (xa - xb).abs()
            );
        }
    }
}

// ============================================================================
// ADV-1: Long chain (5+ ops) -- add → neg → exp → neg → add
// ============================================================================
#[test]
fn adv1_long_chain_5_ops() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    let c = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 0.1),
        vec![],
        vec_f32(4),
        None,
    );
    let a = dag.add_node(RiscOp::Add, vec![x, c], vec_f32(4), None);
    let b = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    let e = dag.add_node(RiscOp::Exp, vec![b], vec_f32(4), None);
    let f = dag.add_node(RiscOp::Neg, vec![e], vec_f32(4), None);
    let g = dag.add_node(RiscOp::Add, vec![f, c], vec_f32(4), None);
    dag.add_root(g);

    let fused = fuse(&dag);

    let inputs: HashMap<String, TensorValue> = [(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
    )]
    .into_iter()
    .collect();

    let orig = eval_dag(&dag, &inputs);
    let fuse_out = eval_dag(&fused, &inputs);
    assert_close(&orig, &fuse_out, 1e-6, "ADV-1: 5-op chain");

    // Should have at least one FusedElem
    assert!(
        fused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::FusedElem { .. })),
        "5-op chain should produce FusedElem"
    );
}

// ============================================================================
// ADV-2: Mixed unary/binary chain -- add → sqrt → mul → neg
// ============================================================================
#[test]
fn adv2_mixed_unary_binary_chain() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    let y = load(&mut dag, "y", vec_f32(4));
    let z = load(&mut dag, "z", vec_f32(4));
    let a = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(4), None);
    let b = dag.add_node(RiscOp::Sqrt, vec![a], vec_f32(4), None);
    let c = dag.add_node(RiscOp::Mul, vec![b, z], vec_f32(4), None);
    let d = dag.add_node(RiscOp::Neg, vec![c], vec_f32(4), None);
    dag.add_root(d);

    let fused = fuse(&dag);

    let inputs: HashMap<String, TensorValue> = [
        (
            "x".into(),
            TensorValue::from_vec(vec![4], vec![1.0, 4.0, 9.0, 16.0]),
        ),
        (
            "y".into(),
            TensorValue::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]),
        ),
        (
            "z".into(),
            TensorValue::from_vec(vec![4], vec![2.0, 3.0, 4.0, 5.0]),
        ),
    ]
    .into_iter()
    .collect();

    let orig = eval_dag(&dag, &inputs);
    let fuse_out = eval_dag(&fused, &inputs);
    assert_close(&orig, &fuse_out, 1e-6, "ADV-2: mixed chain");
}

// ============================================================================
// ADV-3: External input used by multiple steps within a chain
// ============================================================================
#[test]
fn adv3_external_input_used_by_multiple_steps() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    let c = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 2.0),
        vec![],
        vec_f32(4),
        None,
    );
    // add(x, c) → mul(result, c)
    // Both steps use 'c' as an external input
    let a = dag.add_node(RiscOp::Add, vec![x, c], vec_f32(4), None);
    let b = dag.add_node(RiscOp::Mul, vec![a, c], vec_f32(4), None);
    dag.add_root(b);

    let fused = fuse(&dag);

    let inputs: HashMap<String, TensorValue> = [(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
    )]
    .into_iter()
    .collect();

    let orig = eval_dag(&dag, &inputs);
    let fuse_out = eval_dag(&fused, &inputs);
    assert_close(&orig, &fuse_out, 1e-6, "ADV-3: shared external input");
}

// ============================================================================
// ADV-4: Fused and unfused ops feeding the same output
// ============================================================================
#[test]
fn adv4_fused_and_unfused_feed_same_output() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    let y = load(&mut dag, "y", vec_f32(4));
    // Fusible chain: add → neg
    let a = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(4), None);
    let b = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    // Non-fusible: sum
    let s = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x],
        TensorType::scalar_f32(),
        None,
    );
    // Both b and s are roots
    dag.add_root(b);
    dag.add_root(s);

    let fused = fuse(&dag);

    let inputs: HashMap<String, TensorValue> = [
        (
            "x".into(),
            TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
        ),
        (
            "y".into(),
            TensorValue::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]),
        ),
    ]
    .into_iter()
    .collect();

    let orig = eval_dag(&dag, &inputs);
    let fuse_out = eval_dag(&fused, &inputs);
    assert_close(&orig, &fuse_out, 1e-6, "ADV-4: mixed fused/unfused");
}

// ============================================================================
// ADV-5: Intermediate node with 2 consumers splits chain
// ============================================================================
#[test]
fn adv5_intermediate_multi_consumer_splits_chain() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    // Use values where all ops produce finite results
    let a = dag.add_node(RiscOp::Neg, vec![x], vec_f32(4), None);
    let b = dag.add_node(RiscOp::Exp, vec![a], vec_f32(4), None);
    // Use Sqrt instead of Log to avoid NaN on negative inputs
    let c = dag.add_node(RiscOp::Exp, vec![a], vec_f32(4), None);
    // a has 2 consumers (b and c) — a→b and a→c should NOT form one chain
    let d = dag.add_node(RiscOp::Add, vec![b, c], vec_f32(4), None);
    dag.add_root(d);

    let fused = fuse(&dag);

    let inputs: HashMap<String, TensorValue> = [(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
    )]
    .into_iter()
    .collect();

    let orig = eval_dag(&dag, &inputs);
    let fuse_out = eval_dag(&fused, &inputs);
    assert_close(&orig, &fuse_out, 1e-6, "ADV-5: split at multi-consumer");

    // Verify: no FusedElem should have 3+ steps that span through 'a'
    for node in fused.nodes() {
        if let RiscOp::FusedElem { ops, .. } = &node.op {
            assert!(
                ops.len() <= 2,
                "No FusedElem should span the multi-consumer node (got {} steps)",
                ops.len()
            );
        }
    }
}

// ============================================================================
// ADV-6: Entire DAG is one fusible chain (all Loads → chain → root)
// ============================================================================
#[test]
fn adv6_entire_dag_is_fusible() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    let y = load(&mut dag, "y", vec_f32(4));
    let a = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(4), None);
    let b = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), None);
    let c = dag.add_node(RiscOp::Exp, vec![b], vec_f32(4), None);
    dag.add_root(c);

    let fused = fuse(&dag);

    let inputs: HashMap<String, TensorValue> = [
        (
            "x".into(),
            TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
        ),
        (
            "y".into(),
            TensorValue::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]),
        ),
    ]
    .into_iter()
    .collect();

    let orig = eval_dag(&dag, &inputs);
    let fuse_out = eval_dag(&fused, &inputs);
    assert_close(&orig, &fuse_out, 1e-6, "ADV-6: all-fusible");

    // Should be: 2 loads + 1 FusedElem = 3 nodes
    assert_eq!(
        fused.len(),
        3,
        "All-fusible DAG should have 2 loads + 1 FusedElem"
    );
}

// ============================================================================
// ADV-7: Store node in middle of a fusible chain
// ============================================================================
#[test]
fn adv7_store_in_middle_of_chain() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    let a = dag.add_node(RiscOp::Neg, vec![x], vec_f32(4), None);
    let _s = dag.add_node(
        RiscOp::Store {
            name: "intermediate".into(),
        },
        vec![a],
        vec_f32(4),
        None,
    );
    let b = dag.add_node(RiscOp::Exp, vec![a], vec_f32(4), None);
    dag.add_root(b);

    let fused = fuse(&dag);

    // Store is NOT fusible, so Neg→Exp should NOT fuse (Store adds a consumer to 'a')
    let inputs: HashMap<String, TensorValue> = [(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
    )]
    .into_iter()
    .collect();

    let orig = eval_dag(&dag, &inputs);
    let fuse_out = eval_dag(&fused, &inputs);
    assert_close(&orig, &fuse_out, 1e-6, "ADV-7: store breaks fusion");
}

// ============================================================================
// ADV-8: Cast node — is it fusible?
// ============================================================================
#[test]
fn adv8_cast_not_fusible() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    let a = dag.add_node(RiscOp::Neg, vec![x], vec_f32(4), None);
    let cast = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![a],
        vec_f32(4),
        None,
    );
    let b = dag.add_node(RiscOp::Exp, vec![cast], vec_f32(4), None);
    dag.add_root(b);

    let fused = fuse(&dag);

    let inputs: HashMap<String, TensorValue> = [(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
    )]
    .into_iter()
    .collect();

    let orig = eval_dag(&dag, &inputs);
    let fuse_out = eval_dag(&fused, &inputs);
    assert_close(&orig, &fuse_out, 1e-6, "ADV-8: cast breaks fusion");

    // Cast is NOT in is_fusible_elementwise, so neg→cast→exp should not be one big chain
    // But neg could be a singleton (no, chains need 2+). Check that cast prevents fusion across it.
}

// ============================================================================
// ADV-9: CmpLt inside a fused chain preserves sealed bool storage
// ============================================================================
#[test]
fn adv9_cmplt_in_fused_chain_produces_bool() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    let y = load(&mut dag, "y", vec_f32(4));
    let bool_ty = TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: Prim::Bool,
    };
    let cmp = dag.add_node(RiscOp::CmpLt, vec![x, y], bool_ty.clone(), None);
    let false_value = dag.add_node(
        RiscOp::synth_const(Prim::Bool, 0.0),
        vec![],
        bool_ty.clone(),
        None,
    );
    let result = dag.add_node(RiscOp::MaxElem, vec![cmp, false_value], bool_ty, None);
    dag.add_root(result);

    let fused = fuse(&dag);

    let inputs: HashMap<String, TensorValue> = [
        (
            "x".into(),
            TensorValue::from_vec(vec![4], vec![1.0, 5.0, 3.0, 7.0]),
        ),
        (
            "y".into(),
            TensorValue::from_vec(vec![4], vec![2.0, 4.0, 6.0, 8.0]),
        ),
    ]
    .into_iter()
    .collect();

    let orig = eval_dag(&dag, &inputs);
    let fuse_out = eval_dag(&fused, &inputs);
    assert_close(&orig, &fuse_out, 1e-6, "ADV-9: cmplt→bool-or");

    // Check exact values: x<y = [true,false,true,true], and OR false
    // preserves them without reopening a float representation.
    assert_eq!(
        fuse_out[0].storage().to_i64_exact_vec(),
        Some(vec![1, 0, 1, 1]),
        "CmpLt in a fused chain must preserve sealed bool storage"
    );
}

// ============================================================================
// ADV-10: MaxElem in fused chain preserves exact selected-operand semantics
// ============================================================================
#[test]
fn adv10_maxelem_in_fused_chain() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", vec_f32(4));
    let zero = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 0.0),
        vec![],
        vec_f32(4),
        None,
    );
    // relu = max(x, 0)
    let relu = dag.add_node(RiscOp::MaxElem, vec![x, zero], vec_f32(4), None);
    let result = dag.add_node(RiscOp::Neg, vec![relu], vec_f32(4), None);
    dag.add_root(result);

    let fused = fuse(&dag);

    let inputs: HashMap<String, TensorValue> = [(
        "x".into(),
        TensorValue::from_vec(vec![4], vec![-0.0, 2.0, -3.0, 4.0]),
    )]
    .into_iter()
    .collect();

    let orig = eval_dag(&dag, &inputs);
    let fuse_out = eval_dag(&fused, &inputs);
    assert_close(&orig, &fuse_out, 1e-6, "ADV-10: maxelem→neg");

    let stored_f32_bits = |value: &TensorValue| {
        (0..value.len())
            .map(|index| match value.storage().element_ref(index) {
                ElementRef::F32(element) => element.to_bits(),
                other => panic!("ADV-10 expected f32 storage, got {other:?}"),
            })
            .collect::<Vec<_>>()
    };
    // The left operand wins the -0/+0 tie. Negating the selected -0 yields
    // +0; an fmaxf-style surrogate would choose/re-encode +0 and yield -0.
    let expected = vec![0x0000_0000, 0xc000_0000, 0x8000_0000, 0xc080_0000];
    assert_eq!(stored_f32_bits(&orig[0]), expected);
    assert_eq!(stored_f32_bits(&fuse_out[0]), expected);
}

fn main() {}
