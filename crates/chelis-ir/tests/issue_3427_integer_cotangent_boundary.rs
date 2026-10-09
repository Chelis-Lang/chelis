//! chelis#3427: the integer-to-float cast is a zero-cotangent boundary.
//!
//! [04-NUM-14] gives a bool or integer source no cotangent, so an integer
//! computation reached only through a cast is an exact forward value. One that
//! is itself differentiated, because it computes from an integer parameter
//! selected for differentiation, keeps its atom's structural rejection
//! ([05-OP-64]), whichever integer operation it is.
use chelis_ir::dag::{Dag, NodeId, RiscOp, TensorType};
use chelis_ir::grad::{AdError, AdRejectionReason, grad_dag_checked};
use chelis_types::types::Prim;

fn scalar_at(precision: Prim) -> TensorType {
    TensorType {
        dims: vec![],
        precision,
    }
}

const BINARY: [(RiscOp, &str, AdRejectionReason); 8] = [
    (
        RiscOp::Add,
        "add",
        AdRejectionReason::IntegerArithmeticOutput,
    ),
    (
        RiscOp::Sub,
        "sub",
        AdRejectionReason::IntegerArithmeticOutput,
    ),
    (
        RiscOp::Mul,
        "mul",
        AdRejectionReason::IntegerArithmeticOutput,
    ),
    (
        RiscOp::MaxElem,
        "max_elem",
        AdRejectionReason::IntegerArithmeticOutput,
    ),
    (
        RiscOp::MinElem,
        "min_elem",
        AdRejectionReason::IntegerArithmeticOutput,
    ),
    (
        RiscOp::FloorDiv,
        "floor_div",
        AdRejectionReason::PiecewiseConstant,
    ),
    (
        RiscOp::TruncDiv,
        "trunc_div",
        AdRejectionReason::PiecewiseConstant,
    ),
    (RiscOp::Mod, "mod", AdRejectionReason::TruncatedQuotientJump),
];

/// `x * cast(op(left, right), f64)`, returning `(dag, x, left, output)`.
fn coefficient_graph(op: RiscOp) -> (Dag, NodeId, NodeId, NodeId) {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let float_ty = scalar_at(Prim::F64);
    let int_ty = scalar_at(Prim::Int64);
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        float_ty.clone(),
        None,
    );
    let left = dag.add_node(
        decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        int_ty.clone(),
        None,
    );
    let right = dag.add_node(
        decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        int_ty.clone(),
        None,
    );
    let integer = dag.add_node(decl, op, vec![left, right], int_ty, None);
    let coefficient = dag.add_node(
        decl,
        RiscOp::Cast {
            new_precision: Prim::F64,
        },
        vec![integer],
        float_ty.clone(),
        None,
    );
    let output = dag.add_node(decl, RiscOp::Mul, vec![x, coefficient], float_ty, None);
    (dag, x, left, output)
}

#[test]
fn an_integer_coefficient_reached_through_a_cast_is_a_forward_value() {
    for (op, name, _) in BINARY {
        let (dag, x, _, output) = coefficient_graph(op);
        let result = grad_dag_checked(&dag, output, &[x])
            .unwrap_or_else(|error| panic!("{name} coefficient must differentiate: {error}"));
        assert!(result.grad_nodes.contains_key(&x), "{name}");
    }
}

#[test]
fn a_differentiated_integer_parameter_keeps_each_operation_rejection() {
    for (op, name, reason) in BINARY {
        let (dag, x, left, output) = coefficient_graph(op);
        for wrt in [vec![left], vec![x, left]] {
            assert_eq!(
                grad_dag_checked(&dag, output, &wrt).err(),
                Some(AdError::NotSupported {
                    op: name,
                    reason: reason.clone(),
                }),
                "{name} with wrt {wrt:?}"
            );
        }
    }
}
