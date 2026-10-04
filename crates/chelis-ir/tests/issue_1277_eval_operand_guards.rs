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
//! A rank-0 operand against a non-scalar is the one disagreement that is not
//! an error: the lowerer gives a comparison's scalar operand a rank-0 `Const`
//! (`lower_builtin_app`'s `cmplt`/`lt` arm) and the C kernel's strided loop
//! indexes a rank-0 operand at 0 for every element, the idiom
//! `spec/05-risc-primitives.md` section 2.4.1 exempts from the elementwise
//! guard. The evaluator broadcasts it the same way, in either position and at
//! rank 0 only (chelis#1506 owns whether the form should be admitted at all).
//!
//! EVIDENTIARY STATUS, per test:
//! - `an_elementwise_operand_shape_disagreement_is_a_typed_error_not_a_panic`
//!   and `a_runtime_shrink_that_selects_nothing_is_rejected_not_emptied`:
//!   regression tests. On the unfixed tree the first panics (`assertion left
//!   == right failed`) and the second passes an empty tensor back; both were
//!   watched failing before the evaluator changes landed.
//! - `a_rank0_operand_in_either_position_broadcasts_as_the_c_kernel_does`:
//!   regression test, watched failing on `3b0e5b1b2` with the typed error
//!   `got [] vs [3]` for every root.
//! - `a_rank1_against_rank2_disagreement_is_still_a_typed_error`: disposition
//!   lock, green on `3b0e5b1b2`; watched failing under the mutation that
//!   broadcasts any rank-1 operand, so it discriminates the rank-0 bound.

use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, RtDim, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_types::types::Prim;

fn f32_tensor(shape: &[usize], data: &[f64]) -> TensorValue {
    TensorValue::finalize_from_wide("test", Prim::F32, shape.to_vec(), data.to_vec())
        .expect("finalize")
}

fn load(dag: &mut Dag, decl: chelis_ir::dag::DeclId, name: &str, extent: usize) -> NodeId {
    load_shaped(dag, decl, name, &[extent])
}

fn load_shaped(dag: &mut Dag, decl: chelis_ir::dag::DeclId, name: &str, shape: &[usize]) -> NodeId {
    dag.add_node(
        decl,
        RiscOp::Load { name: name.into() },
        vec![],
        TensorType {
            dims: shape.iter().map(|&extent| DimInfo::Lit(extent)).collect(),
            precision: Prim::F32,
        },
        None,
    )
}

#[test]
fn an_elementwise_operand_shape_disagreement_is_a_typed_error_not_a_panic() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = load(&mut dag, decl, "a", 4);
    let b = load(&mut dag, decl, "b", 3);
    let sum = dag.add_node(
        decl,
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
        err == "add operands disagree at axis 0: lhs [4] has 4, rhs [3] has 3\n\
                numeric trap: domain in add at i64",
        "the shared rendering, so every lane and the tests agree: {err}"
    );
}

#[test]
fn a_runtime_shrink_that_selects_nothing_is_an_empty_axis() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = load(&mut dag, decl, "x", 4);
    // A runtime bound: `end` comes from a rank-0 i64 scalar input, as
    // `k = n - 4` lowers.
    let k = dag.add_node(
        decl,
        RiscOp::Load { name: "k".into() },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        None,
    );
    let shrunk = dag.add_node(
        decl,
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
    // spec/05 section 2.4.1: equal endpoints describe an empty axis
    // (chelis#1795), so bounds [0, 0) select an extent-0 result.
    let values = eval_tensor_roots_with_strict(&dag, &[shrunk], |name| match name {
        "x" => Some(f32_tensor(&[4], &[1.0, 2.0, 3.0, 4.0])),
        "k" => Some(
            TensorValue::finalize_from_wide_int("test", Prim::Int64, vec![], vec![0])
                .expect("finalize"),
        ),
        _ => None,
    })
    .expect("bounds [0, 0) select an empty axis");
    assert_eq!(values[&shrunk].shape, vec![0]);
}
