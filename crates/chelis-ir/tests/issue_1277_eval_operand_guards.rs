//! chelis#1277 Slice B2h: operand checks the DAG evaluator owes as typed
//! errors once the host interpreter routes def applications through it.
//!
//! Before the routing, `chelis eval` reached `chelis_ir::eval` only for
//! Tensor-lane roots and transforms, so a runtime operand disagreement inside
//! a kernel was either impossible (the lowerer had proved the shapes) or an
//! internal invariant. Through the routing it is user input: an elementwise
//! op whose operands disagree at run time, or a `shrink` whose runtime bounds
//! select nothing. The interpreter reported both as errors with the phrases
//! asserted below; the C runtime guards both (chelis#664, chelis#616). The DAG
//! evaluator asserted on the first and returned an empty tensor for the
//! second.
//!
//! EVIDENTIARY STATUS: regression tests. On the unfixed tree the first panics
//! (`assertion left == right failed`) and the second passes an empty tensor
//! back; both were watched failing before the evaluator changes landed.

use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, RtDim, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_types::types::Prim;

fn f32_tensor(shape: &[usize], data: &[f64]) -> TensorValue {
    TensorValue::finalize_from_wide("test", Prim::F32, shape.to_vec(), data.to_vec())
        .expect("finalize")
}

fn load(dag: &mut Dag, name: &str, extent: usize) -> NodeId {
    dag.add_node(
        RiscOp::Load { name: name.into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(extent)],
            precision: Prim::F32,
        },
        None,
    )
}

#[test]
fn an_elementwise_operand_shape_disagreement_is_a_typed_error_not_a_panic() {
    let mut dag = Dag::new();
    let a = load(&mut dag, "a", 4);
    let b = load(&mut dag, "b", 3);
    let sum = dag.add_node(
        RiscOp::Add,
        vec![a, b],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(sum);
    let err = eval_tensor_roots_with_strict(&dag, &[sum], |name| match name {
        "a" => Some(f32_tensor(&[4], &[1.0, 2.0, 3.0, 4.0])),
        "b" => Some(f32_tensor(&[3], &[1.0, 2.0, 3.0])),
        _ => None,
    })
    .expect_err("[4] against [3] must be rejected");
    assert!(
        err.contains("tensor shapes must match for elementwise op, got [4] vs [3]"),
        "the interpreter's phrase, so both eval paths and the tests agree: {err}"
    );
}

#[test]
fn a_runtime_shrink_that_selects_nothing_is_rejected_not_emptied() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", 4);
    // A runtime bound: `end` comes from a rank-0 int64 scalar input, as
    // `k = n - 4` lowers.
    let k = dag.add_node(
        RiscOp::Load { name: "k".into() },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        None,
    );
    let shrunk = dag.add_node(
        RiscOp::Shrink {
            bounds: vec![(RtDim::Lit(0), RtDim::Node(1))],
        },
        vec![x, k],
        TensorType {
            dims: vec![DimInfo::Named("u".into(), None)],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(shrunk);
    let err = eval_tensor_roots_with_strict(&dag, &[shrunk], |name| match name {
        "x" => Some(f32_tensor(&[4], &[1.0, 2.0, 3.0, 4.0])),
        "k" => Some(
            TensorValue::finalize_from_wide_int("test", Prim::Int64, vec![], vec![0])
                .expect("finalize"),
        ),
        _ => None,
    })
    .expect_err("bounds [0, 0) select nothing and must be rejected");
    assert!(
        err.contains("is empty or inverted"),
        "the interpreter's phrase for an empty bound: {err}"
    );
}
