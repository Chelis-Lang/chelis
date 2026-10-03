//! chelis#2957 GPU fence (`spec/design/correctly_rounded_math.md` §4.3): the
//! hip lane has no correctly rounded transcendental kernels and does not
//! establish a correctly rounded `sqrt`, so codegen rejects every [05-OP-46]
//! transcendental and `sqrt` through [05-UNS-1] instead of computing them
//! with a vendor library. Exact operations stay admitted.

mod support;

use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use chelis_types::unsupported::Unsupported;

fn f32_vec() -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: Prim::F32,
    }
}

/// `out = op(add(a, b))`: the add feeds the unary op so the elementwise
/// chain is eligible for fusion.
fn chain_dag(op: RiscOp) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        f32_vec(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        f32_vec(),
        None,
    );
    let sum = dag.add_node(decl, RiscOp::Add, vec![a, b], f32_vec(), None);
    let applied = dag.add_node(decl, op, vec![sum], f32_vec(), None);
    let stored = dag.add_node(
        decl,
        RiscOp::Store { name: "out".into() },
        vec![applied],
        f32_vec(),
        None,
    );
    dag.add_root(stored);
    dag
}

fn codegen(dag: &Dag, name: &str) -> Result<(), Unsupported> {
    support::codegen_hip(dag, name).map(|_| ())
}

#[test]
fn every_transcendental_and_sqrt_is_rejected_with_a_typed_diagnostic() {
    for (op, name) in [
        (RiscOp::Exp, "exp"),
        (RiscOp::Log, "log"),
        (RiscOp::Sin, "sin"),
        (RiscOp::Cos, "cos"),
        (RiscOp::Tan, "tan"),
        (RiscOp::Atan, "atan"),
        (RiscOp::Tanh, "tanh"),
        (RiscOp::Sqrt, "sqrt"),
    ] {
        let error = codegen(&chain_dag(op), name).expect_err(name);
        let rendered = error.to_string();
        assert!(rendered.contains("[05-OP-46]"), "{name}: {rendered}");
        assert!(rendered.contains("codegen:hip"), "{name}: {rendered}");
        assert!(
            rendered.contains(&format!("`{name}`")),
            "{name}: {rendered}"
        );
    }
}

#[test]
fn exact_operations_are_admitted() {
    for (op, name) in [(RiscOp::Neg, "neg"), (RiscOp::Abs, "abs")] {
        codegen(&chain_dag(op), name).unwrap_or_else(|error| panic!("{name}: {error}"));
    }
}

#[test]
fn a_transcendental_inside_a_fused_chain_is_rejected() {
    let fused = chelis_ir::fuse::fuse(&chain_dag(RiscOp::Tanh));
    assert!(
        fused
            .nodes()
            .iter()
            .any(|node| matches!(&node.op, RiscOp::FusedElem { .. }))
            && !fused
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Tanh)),
        "the witness must reach codegen as a fused chain, not a bare tanh node"
    );
    let error = codegen(&fused, "fused_tanh").expect_err("fused tanh");
    assert!(error.to_string().contains("`tanh`"), "{error}");
}

#[test]
fn sqrt_inside_a_fused_chain_is_rejected() {
    let fused = chelis_ir::fuse::fuse(&chain_dag(RiscOp::Sqrt));
    assert!(
        fused
            .nodes()
            .iter()
            .any(|node| matches!(&node.op, RiscOp::FusedElem { .. }))
            && !fused
                .nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Sqrt)),
        "the witness must reach codegen as a fused chain, not a bare sqrt node"
    );
    let error = codegen(&fused, "fused_sqrt").expect_err("fused sqrt");
    assert!(error.to_string().contains("`sqrt`"), "{error}");
}
