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
use chelis_types::{RawScalar, finalize_scalar};

fn f32_tensor(shape: &[usize], data: &[f64]) -> TensorValue {
    TensorValue::finalize_from_wide("test", Prim::F32, shape.to_vec(), data.to_vec())
        .expect("finalize")
}

fn load(dag: &mut Dag, name: &str, extent: usize) -> NodeId {
    load_shaped(dag, name, &[extent])
}

fn load_shaped(dag: &mut Dag, name: &str, shape: &[usize]) -> NodeId {
    dag.add_node(
        RiscOp::Load { name: name.into() },
        vec![],
        TensorType {
            dims: shape.iter().map(|&extent| DimInfo::Lit(extent)).collect(),
            precision: Prim::F32,
        },
        None,
    )
}

/// A rank-0 f32 `Const`, as the lowerer builds one for a scalar literal
/// operand.
fn scalar_const(dag: &mut Dag, value: f64) -> NodeId {
    let value = finalize_scalar("test", Prim::F32, RawScalar::Float(value)).expect("finalize");
    dag.add_node(
        RiscOp::Const { value },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    )
}

fn binary(dag: &mut Dag, op: RiscOp, lhs: NodeId, rhs: NodeId, prim: Prim) -> NodeId {
    let node = dag.add_node(
        op,
        vec![lhs, rhs],
        TensorType {
            dims: vec![DimInfo::Lit(3)],
            precision: prim,
        },
        None,
    );
    dag.add_root(node);
    node
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

/// `coral_prerequisites.rs`'s `above`, `lt_right` and an arithmetic sibling,
/// as the lowerer emits them: `gt(x, 1.5)` is `CmpLt(Const 1.5, x)`, `lt(x,
/// 2.5)` is `CmpLt(x, Const 2.5)`. The expected values are what the compiled
/// C kernel prints for the same program.
#[test]
fn a_rank0_operand_in_either_position_broadcasts_as_the_c_kernel_does() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x", 3);
    let above_bound = scalar_const(&mut dag, 1.5);
    let above = binary(&mut dag, RiscOp::CmpLt, above_bound, x, Prim::Bool);
    let lt_bound = scalar_const(&mut dag, 2.5);
    let lt_right = binary(&mut dag, RiscOp::CmpLt, x, lt_bound, Prim::Bool);
    let shift = scalar_const(&mut dag, 1.5);
    let shifted = binary(&mut dag, RiscOp::Add, x, shift, Prim::F32);
    let values =
        eval_tensor_roots_with_strict(&dag, &[above, lt_right, shifted], |name| match name {
            "x" => Some(f32_tensor(&[3], &[1.0, 2.0, 3.0])),
            _ => None,
        })
        .unwrap_or_else(|err| panic!("a rank-0 operand broadcasts as the C loop does: {err}"));
    for (root, expected) in [
        (above, vec![0.0, 1.0, 1.0]),
        (lt_right, vec![1.0, 1.0, 0.0]),
        (shifted, vec![2.5, 3.5, 4.5]),
    ] {
        let value = &values[&root];
        assert_eq!(value.shape, vec![3], "the non-scalar operand's shape");
        assert_eq!(value.to_f64_lossy_vec(), expected);
    }
}

/// The bound is rank 0, not "the smaller operand": `[3]` against `[2, 3]`
/// keeps the typed error the C runtime guard and the interpreter report.
#[test]
fn a_rank1_against_rank2_disagreement_is_still_a_typed_error() {
    let mut dag = Dag::new();
    let row = load(&mut dag, "row", 3);
    let grid = load_shaped(&mut dag, "grid", &[2, 3]);
    let sum = dag.add_node(
        RiscOp::Add,
        vec![row, grid],
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(sum);
    let err = eval_tensor_roots_with_strict(&dag, &[sum], |name| match name {
        "row" => Some(f32_tensor(&[3], &[1.0, 2.0, 3.0])),
        "grid" => Some(f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        _ => None,
    })
    .expect_err("[3] against [2, 3] is not a rank-0 broadcast and must be rejected");
    assert!(
        err.contains("tensor shapes must match for elementwise op, got [3] vs [2, 3]"),
        "{err}"
    );
}
