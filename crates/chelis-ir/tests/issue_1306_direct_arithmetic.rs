use chelis_ir::dag::{Dag, RiscOp, TensorType};
use chelis_ir::{tier2, verify};

fn scalar_f32() -> TensorType {
    TensorType::scalar_f32()
}

#[test]
fn sub_lowers_to_one_direct_identity_without_synthetic_negation() {
    let mut dag = Dag::new();
    let ty = scalar_f32();
    let left = dag.add_node(
        RiscOp::synth_const(ty.precision, -1.0),
        vec![],
        ty.clone(),
        None,
    );
    let right = dag.add_node(
        RiscOp::synth_const(ty.precision, 2.0),
        vec![],
        ty.clone(),
        None,
    );

    let result = tier2::lower_sub(&mut dag, left, right, &ty, Some("sub.expr"));

    assert!(verify::verify(&dag).is_empty());
    assert_eq!(dag.len(), 3, "direct sub adds exactly one node");
    assert_eq!(dag.get(result).unwrap().op, RiscOp::Sub);
    assert_eq!(dag.get(result).unwrap().inputs, vec![left, right]);
    assert_eq!(
        dag.get(result).unwrap().span_id.as_deref(),
        Some("sub.expr")
    );
    assert!(
        dag.nodes()
            .iter()
            .all(|node| !matches!(node.op, RiscOp::Neg | RiscOp::Add)),
        "sub must not introduce add or neg"
    );
}

#[test]
fn min_elem_lowers_to_one_direct_selection_without_arithmetic_surrogate() {
    let mut dag = Dag::new();
    let ty = scalar_f32();
    let left = dag.add_node(
        RiscOp::synth_const(ty.precision, -1.0),
        vec![],
        ty.clone(),
        None,
    );
    let right = dag.add_node(
        RiscOp::synth_const(ty.precision, 2.0),
        vec![],
        ty.clone(),
        None,
    );

    let result = tier2::lower_min_elem(&mut dag, left, right, &ty, Some("min.expr"));

    assert!(verify::verify(&dag).is_empty());
    assert_eq!(dag.len(), 3, "direct min_elem adds exactly one node");
    assert_eq!(dag.get(result).unwrap().op, RiscOp::MinElem);
    assert_eq!(dag.get(result).unwrap().inputs, vec![left, right]);
    assert_eq!(
        dag.get(result).unwrap().span_id.as_deref(),
        Some("min.expr")
    );
    assert!(
        dag.nodes()
            .iter()
            .all(|node| !matches!(node.op, RiscOp::Neg | RiscOp::MaxElem)),
        "min_elem must not introduce neg or max_elem"
    );
}
