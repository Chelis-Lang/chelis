use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_ir::vmap::vectorize_axis0;
use chelis_types::types::Prim;
use std::collections::HashMap;

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn mat_f32(m: usize, n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(m), DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn eval_root(dag: &Dag, inputs: &HashMap<String, TensorValue>) -> TensorValue {
    let root = dag.roots()[0];
    let values = eval_tensor_roots_with_strict(dag, &[root], |name| inputs.get(name).cloned())
        .expect("evaluation should succeed");
    values[&root].clone()
}

#[test]
fn vmap_elementwise_vectorizes_axis_zero() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(3), None);
    let y = dag.add_node(RiscOp::Neg, vec![x], vec_f32(3), None);
    dag.add_root(y);

    let vmapped = vectorize_axis0(&dag, DimInfo::Lit(2)).expect("vmap should succeed");
    let value = eval_root(
        &vmapped,
        &HashMap::from([(
            "x".to_string(),
            TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, -5.0, 6.0]),
        )]),
    );
    assert_eq!(value.shape, vec![2, 3]);
    assert_eq!(value.data, vec![-1.0, -2.0, -3.0, -4.0, 5.0, -6.0]);
}

#[test]
fn vmap_reduction_shifts_the_reduced_axis() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(2, 3),
        None,
    );
    let y = dag.add_node(RiscOp::Sum { axis: 1 }, vec![x], vec_f32(2), None);
    dag.add_root(y);

    let vmapped = vectorize_axis0(&dag, DimInfo::Lit(2)).expect("vmap should succeed");
    let value = eval_root(
        &vmapped,
        &HashMap::from([(
            "x".to_string(),
            TensorValue::from_vec(
                vec![2, 2, 3],
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 7.0, 8.0, 9.0,
                ],
            ),
        )]),
    );
    assert_eq!(value.shape, vec![2, 2]);
    assert_eq!(value.data, vec![6.0, 15.0, 60.0, 24.0]);
}

#[test]
fn vmap_nested_adds_multiple_batch_axes() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
    dag.add_root(x);

    let inner = vectorize_axis0(&dag, DimInfo::Lit(3)).expect("inner vmap should succeed");
    let outer = vectorize_axis0(&inner, DimInfo::Lit(2)).expect("outer vmap should succeed");
    let root = outer.roots()[0];
    let node = outer.get(root).expect("root");
    assert_eq!(
        node.output_type,
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        }
    );
}

#[test]
fn vmap_batched_matmul_stays_in_expand_mul_sum_form() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_f32(2, 3),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_f32(3, 4),
        None,
    );
    let a_exp = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        },
        None,
    );
    let b_exp = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(2),
        },
        vec![b],
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        },
        None,
    );
    let prod = dag.add_node(
        RiscOp::Mul,
        vec![a_exp, b_exp],
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        },
        None,
    );
    let out = dag.add_node(RiscOp::Sum { axis: 1 }, vec![prod], mat_f32(2, 4), None);
    dag.add_root(out);

    let vmapped = vectorize_axis0(&dag, DimInfo::Lit(5)).expect("vmap should succeed");
    assert!(
        vmapped
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::Expand { axis: 3, .. })),
        "batched matmul should stay in the generic expand->mul->sum decomposition"
    );
}
