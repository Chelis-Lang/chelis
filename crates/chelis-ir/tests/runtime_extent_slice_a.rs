//! Slice A executable contract for chelis#1277.

use chelis_ir::dag::{Dag, DimInfo, RiscOp, RtAxis, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_ir::verify;
use chelis_ir::vmap::vectorize_axis0;
use chelis_types::types::Prim;
use chelis_unord::UnordMap;

fn ty(dims: Vec<DimInfo>, precision: Prim) -> TensorType {
    TensorType { dims, precision }
}

fn scalar(precision: Prim) -> TensorType {
    ty(vec![], precision)
}

#[test]
fn input_axis_expand_verifies_and_evaluates_from_shape_metadata() {
    let mut dag = Dag::new();
    let value = dag.add_node(
        RiscOp::Load {
            name: "value".into(),
        },
        vec![],
        scalar(Prim::F32),
        None,
    );
    let source = dag.add_node(
        RiscOp::Load {
            name: "source".into(),
        },
        vec![],
        ty(vec![DimInfo::Named("n".into(), None)], Prim::F32),
        None,
    );
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![value, source],
        ty(vec![DimInfo::Named("n".into(), None)], Prim::F32),
        None,
    );
    dag.add_root(expanded);

    assert!(verify::verify(&dag).is_empty());
    let values = eval_tensor_roots_with_strict(&dag, &[expanded], |name| match name {
        "value" => Some(TensorValue::from_vec(vec![], vec![2.0])),
        "source" => Some(TensorValue::from_vec(vec![3], vec![4.0, 5.0, 6.0])),
        _ => None,
    })
    .expect("InputAxis expansion must evaluate");
    assert_eq!(values[&expanded].shape, vec![3]);
    assert_eq!(values[&expanded].to_f64_lossy_vec(), vec![2.0; 3]);
}

#[test]
fn node_expand_verifies_and_evaluates_from_int64_scalar_input() {
    let mut dag = Dag::new();
    let value = dag.add_node(
        RiscOp::Load {
            name: "value".into(),
        },
        vec![],
        scalar(Prim::F32),
        None,
    );
    let extent = dag.add_node(
        RiscOp::Load {
            name: "extent".into(),
        },
        vec![],
        scalar(Prim::Int64),
        None,
    );
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Node(1),
        },
        vec![value, extent],
        ty(vec![DimInfo::Named("n".into(), None)], Prim::F32),
        None,
    );
    dag.add_root(expanded);

    assert!(verify::verify(&dag).is_empty());
    let values = eval_tensor_roots_with_strict(&dag, &[expanded], |name| match name {
        "value" => Some(TensorValue::from_vec(vec![], vec![7.0])),
        "extent" => Some(TensorValue::from_vec(vec![], vec![3.0])),
        _ => None,
    })
    .expect("node-valued expansion must evaluate");
    assert_eq!(values[&expanded].shape, vec![3]);
    assert_eq!(values[&expanded].to_f64_lossy_vec(), vec![7.0; 3]);
}

#[test]
fn zero_literal_expand_is_valid_and_empty() {
    let mut dag = Dag::new();
    let value = dag.add_node(
        RiscOp::Load {
            name: "value".into(),
        },
        vec![],
        scalar(Prim::F32),
        None,
    );
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(0),
        },
        vec![value],
        ty(vec![DimInfo::Lit(0)], Prim::F32),
        None,
    );
    dag.add_root(expanded);

    assert!(verify::verify(&dag).is_empty());
    let values = eval_tensor_roots_with_strict(&dag, &[expanded], |name| {
        (name == "value").then(|| TensorValue::from_vec(vec![], vec![1.0]))
    })
    .expect("zero expansion must evaluate");
    assert_eq!(values[&expanded].shape, vec![0]);
    assert!(values[&expanded].is_empty());
}

#[test]
fn input_axis_owner_and_slot_validation_fail_closed() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        ty(vec![DimInfo::Lit(3)], Prim::F32),
        None,
    );
    dag.add_node(
        RiscOp::zero_pad(
            Prim::F32,
            vec![(
                chelis_ir::dag::RtDim::InputAxis {
                    tensor: 1,
                    axis: RtAxis::Lit(0),
                },
                chelis_ir::dag::RtDim::Lit(0),
            )],
        ),
        vec![x],
        ty(vec![DimInfo::Lit(3)], Prim::F32),
        None,
    );
    let errors = verify::verify(&dag);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("InputAxis") && error.contains("Pad")),
        "forbidden owner and missing slot must fail closed: {errors:?}"
    );
}

#[test]
fn movement_ops_reject_unowned_runtime_extent_inputs() {
    let mut dag = Dag::new();
    let value = dag.add_node(
        RiscOp::Load {
            name: "value".into(),
        },
        vec![],
        ty(vec![DimInfo::Lit(3)], Prim::F32),
        None,
    );
    let extent = dag.add_node(
        RiscOp::Load {
            name: "extent".into(),
        },
        vec![],
        scalar(Prim::Int64),
        None,
    );
    let unowned = dag.add_node(
        RiscOp::Load {
            name: "unowned".into(),
        },
        vec![],
        scalar(Prim::Int64),
        None,
    );
    dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![chelis_ir::dag::RtDim::Node(1)],
        },
        vec![value, extent, unowned],
        ty(vec![DimInfo::Lit(3)], Prim::F32),
        None,
    );

    let errors = verify::verify(&dag);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("unowned runtime extent input slot 2")),
        "every non-data movement input must be owned by an RtDim: {errors:?}"
    );
}

#[test]
fn vmap_keeps_shape_bound_shared_and_shifts_its_axis() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        ty(vec![DimInfo::Lit(3)], Prim::F32),
        None,
    );
    let end = dag.add_node(
        RiscOp::Shape { axis: 0 },
        vec![x],
        scalar(Prim::Int64),
        None,
    );
    let y = dag.add_node(
        RiscOp::Shrink {
            bounds: vec![(
                chelis_ir::dag::RtDim::Lit(1),
                chelis_ir::dag::RtDim::Node(1),
            )],
        },
        vec![x, end],
        ty(vec![DimInfo::Lit(2)], Prim::F32),
        None,
    );
    dag.add_root(y);

    let vmapped = vectorize_axis0(&dag, DimInfo::Named("batch".into(), None))
        .expect("vmap must accept a shape bound under a runtime batch extent");
    assert_eq!(vmapped.get(end).unwrap().output_type.dims, vec![]);
    assert!(matches!(
        vmapped.get(end).unwrap().op,
        RiscOp::Shape { axis: 1 }
    ));
    assert!(verify::verify(&vmapped).is_empty());

    let root = vmapped.roots()[0];
    let values = eval_tensor_roots_with_strict(&vmapped, &[root], |name| {
        (name == "x").then(|| TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    })
    .expect("vmapped shape bound must evaluate");
    assert_eq!(values[&root].shape, vec![2, 2]);
    assert_eq!(values[&root].to_f64_lossy_vec(), vec![2.0, 3.0, 5.0, 6.0]);
}

#[test]
fn vmap_shares_shape_extent_across_bound_and_ordinary_uses() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        ty(vec![DimInfo::Lit(3)], Prim::F32),
        None,
    );
    let extent = dag.add_node(
        RiscOp::Shape { axis: 0 },
        vec![x],
        scalar(Prim::Int64),
        None,
    );
    let y = dag.add_node(
        RiscOp::Shrink {
            bounds: vec![(
                chelis_ir::dag::RtDim::Lit(0),
                chelis_ir::dag::RtDim::Node(1),
            )],
        },
        vec![x, extent],
        ty(vec![DimInfo::Lit(3)], Prim::F32),
        None,
    );
    let ordinary_extent = dag.add_node(RiscOp::Copy, vec![extent], scalar(Prim::Int64), None);
    dag.add_root(y);
    dag.add_root(ordinary_extent);

    let vmapped = vectorize_axis0(&dag, DimInfo::Named("batch".into(), None))
        .expect("one shared shape value must serve its bound and ordinary uses");
    assert!(verify::verify(&vmapped).is_empty());
    assert_eq!(
        vmapped
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Shape { .. }))
            .count(),
        1,
        "the shared extent operation must still execute once"
    );

    let roots = vmapped.roots();
    let values = eval_tensor_roots_with_strict(&vmapped, roots, |name| {
        (name == "x").then(|| TensorValue::from_vec(vec![2, 3], vec![1.0; 6]))
    })
    .expect("vmapped shared extent must evaluate");
    assert_eq!(values[&roots[0]].shape, vec![2, 3]);
    assert_eq!(values[&roots[1]].shape, vec![2]);
    assert_eq!(values[&roots[1]].to_f64_lossy_vec(), vec![3.0, 3.0]);
}

#[test]
fn vmap_rejects_element_derived_extent() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        ty(vec![DimInfo::Lit(3)], Prim::F32),
        None,
    );
    let extent = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![x],
        scalar(Prim::F32),
        None,
    );
    let y = dag.add_node(
        RiscOp::Shrink {
            bounds: vec![(
                chelis_ir::dag::RtDim::Lit(0),
                chelis_ir::dag::RtDim::Node(1),
            )],
        },
        vec![x, extent],
        ty(vec![DimInfo::Named("m".into(), None)], Prim::F32),
        None,
    );
    dag.add_root(y);

    let error = vectorize_axis0(&dag, DimInfo::Lit(2)).expect_err("ragged extent must reject");
    assert!(
        error.contains("batch_varying_extent"),
        "unexpected error: {error}"
    );
}

#[test]
fn input_axis_vmap_shifts_literal_axis() {
    let mut dag = Dag::new();
    let value = dag.add_node(
        RiscOp::Load {
            name: "value".into(),
        },
        vec![],
        scalar(Prim::F32),
        None,
    );
    let source = dag.add_node(
        RiscOp::Load {
            name: "source".into(),
        },
        vec![],
        ty(vec![DimInfo::Lit(3)], Prim::F32),
        None,
    );
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![value, source],
        ty(vec![DimInfo::Lit(3)], Prim::F32),
        None,
    );
    dag.add_root(expanded);

    let vmapped = vectorize_axis0(&dag, DimInfo::Lit(2)).expect("vmap InputAxis");
    assert!(matches!(
        vmapped.get(expanded).unwrap().op,
        RiscOp::Expand {
            size: chelis_ir::dag::RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(1)
            },
            ..
        }
    ));
}

#[test]
fn input_axis_eval_does_not_read_tensor_elements() {
    let mut a = UnordMap::new();
    a.insert(
        "source",
        TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
    );
    let mut b = UnordMap::new();
    b.insert(
        "source",
        TensorValue::from_vec(vec![3], vec![9.0, 8.0, 7.0]),
    );
    assert_eq!(a["source"].shape, b["source"].shape);
}
