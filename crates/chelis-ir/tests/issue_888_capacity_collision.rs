//! [04-SHAPE-1] shared-plan outcome for the original #888 collision.
//! The key-level witness lives in capacity_key::tests, where construction is
//! private. No production compatibility normalizer survives just for a test.

use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::ownership::{lower_dag_ownership, plan_c_storage, verify_ownership};
use chelis_types::types::Prim;

fn shape(first: usize, second: usize) -> TensorType {
    TensorType {
        dims: vec![
            DimInfo::Named("n".into(), None),
            DimInfo::Lit(first),
            DimInfo::Lit(second),
        ],
        precision: Prim::F32,
    }
}

fn slots_for(final_type: TensorType) -> usize {
    let mut dag = Dag::new();
    let small = shape(1 << 40, 1 << 40);
    let a = dag.add_node(
        RiscOp::synth_const(Prim::F32, 1.0),
        vec![],
        small.clone(),
        None,
    );
    let b = dag.add_node(RiscOp::Neg, vec![a], small, None);
    let c = dag.add_node(RiscOp::Neg, vec![b], final_type, None);
    dag.add_root(c);
    let program = verify_ownership(lower_dag_ownership(dag).unwrap()).unwrap();
    plan_c_storage(program).unwrap().slots().len()
}

#[test]
fn distinct_large_products_do_not_reuse_storage() {
    assert_eq!(slots_for(shape(1 << 40, 1 << 41)), 3);
}

#[test]
fn equivalent_large_factorizations_still_reuse_storage() {
    assert_eq!(slots_for(shape(1 << 39, 1 << 41)), 2);
}
