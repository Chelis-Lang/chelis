//! chelis#558 / chelis#513: `RiscOp::Shape`, a runtime shape-extraction
//! DAG node. Unit + pipeline coverage for the value node itself: lowering,
//! verification (positive + negative), evaluation, and AD-transparency.

use std::collections::HashMap;

use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_with};
use chelis_ir::grad::grad_dag_checked;
use chelis_ir::verify;
use chelis_types::types::Prim;

// End-to-end lowering (`shape(x, axis)` value -> `RiscOp::Shape` -> eval / C
// backend) is covered by the CLI integration suite in
// `crates/chelis-cli/tests/issue_558_shape_value_read.rs`; this file covers
// the value node's IR-level semantics directly.

fn tensor_ty(dims: Vec<DimInfo>, precision: Prim) -> TensorType {
    TensorType { dims, precision }
}

fn scalar_int_ty(precision: Prim) -> TensorType {
    TensorType {
        dims: vec![],
        precision,
    }
}

// ---------------------------------------------------------------------------
// Evaluation: the node reads the input's runtime extent along `axis`.
// ---------------------------------------------------------------------------

#[test]
fn shape_node_evaluates_to_runtime_extent() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_ty(
            vec![DimInfo::Named("batch".into(), None), DimInfo::Lit(3)],
            Prim::F32,
        ),
        None,
    );
    let s0 = dag.add_node(
        RiscOp::Shape { axis: 0 },
        vec![x],
        scalar_int_ty(Prim::Int32),
        None,
    );
    let s1 = dag.add_node(
        RiscOp::Shape { axis: 1 },
        vec![x],
        scalar_int_ty(Prim::Int32),
        None,
    );

    // Feed a concrete 2x3 input; extents are 2 and 3.
    let mut inputs: HashMap<String, TensorValue> = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    let values = eval_tensor_with(&dag, |name| inputs.get(name).cloned()).expect("eval");
    assert_eq!(values[&s0].shape, Vec::<usize>::new(), "extent is a scalar");
    assert_eq!(
        values[&s0].to_f64_lossy_vec(),
        vec![2.0],
        "axis 0 extent == 2"
    );
    assert_eq!(
        values[&s1].to_f64_lossy_vec(),
        vec![3.0],
        "axis 1 extent == 3"
    );
}

// ---------------------------------------------------------------------------
// Verification negatives: a malformed Shape node fails loud.
// ---------------------------------------------------------------------------

fn verify_shape(input_dims: Vec<DimInfo>, axis: usize, out_ty: TensorType) -> Vec<String> {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_ty(input_dims, Prim::F32),
        None,
    );
    dag.add_node(RiscOp::Shape { axis }, vec![x], out_ty, None);
    verify::verify(&dag)
}

#[test]
fn shape_node_verify_rejects_out_of_range_axis() {
    let errs = verify_shape(vec![DimInfo::Lit(4)], 3, scalar_int_ty(Prim::Int32));
    assert!(
        errs.iter()
            .any(|e| e.contains("shape read") && e.contains("axis")),
        "out-of-range axis must be rejected; errs = {errs:?}"
    );
}

#[test]
fn shape_node_verify_rejects_non_scalar_output() {
    let errs = verify_shape(
        vec![DimInfo::Lit(4)],
        0,
        tensor_ty(vec![DimInfo::Lit(1)], Prim::Int32),
    );
    assert!(
        errs.iter().any(|e| e.contains("rank-0 scalar")),
        "non-scalar output must be rejected; errs = {errs:?}"
    );
}

#[test]
fn shape_node_verify_rejects_non_integer_output() {
    let errs = verify_shape(vec![DimInfo::Lit(4)], 0, scalar_int_ty(Prim::F32));
    assert!(
        errs.iter().any(|e| e.contains("integer scalar")),
        "non-integer output must be rejected; errs = {errs:?}"
    );
}

#[test]
fn shape_node_verify_accepts_wellformed() {
    let errs = verify_shape(
        vec![DimInfo::Named("batch".into(), None), DimInfo::Lit(3)],
        0,
        scalar_int_ty(Prim::Int64),
    );
    assert!(
        errs.is_empty(),
        "well-formed Shape must verify clean; errs = {errs:?}"
    );
}

// ---------------------------------------------------------------------------
// AD-transparency: `grad` accepts a Shape node (it is a trivial constant,
// NOT rejected like `floor_div`), and the adjoint to the input is a zero
// Const (the extent does not depend on the element values).
// ---------------------------------------------------------------------------

#[test]
fn shape_node_is_ad_transparent_with_zero_adjoint() {
    // forward: loss = cast(shape(x, 0), f32)  (a scalar float).
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_ty(vec![DimInfo::Lit(3), DimInfo::Lit(2)], Prim::F32),
        None,
    );
    let sh = dag.add_node(
        RiscOp::Shape { axis: 0 },
        vec![x],
        scalar_int_ty(Prim::Int32),
        None,
    );
    let loss = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![sh],
        scalar_int_ty(Prim::F32),
        None,
    );

    let result = grad_dag_checked(&dag, loss, &[x])
        .expect("grad must accept a Shape node (trivial constant, not rejected)");
    // The gradient DAG must contain the input's zero cotangent: a Const 0.0.
    let has_zero_const = result
        .dag
        .nodes()
        .iter()
        .any(|n| matches!(&n.op, RiscOp::Const { value } if value.as_f64_lossy() == 0.0));
    assert!(
        has_zero_const,
        "Shape adjoint must route a zero cotangent to the input"
    );
}
