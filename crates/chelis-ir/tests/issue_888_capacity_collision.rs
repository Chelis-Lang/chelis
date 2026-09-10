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

/// [04-SHAPE-1]: an expired owned slot needs both exact capacity and exact
/// representation. Unlike a destructive Drop, ordinary last use permits reuse,
/// so removing the representation check must change this plan.
#[test]
fn expired_owned_slots_require_exact_representation_in_both_lanes() {
    use chelis_ir::ownership::plan_hip_storage;
    let dtypes = [
        Prim::F64,
        Prim::F32,
        Prim::F16,
        Prim::Bf16,
        Prim::Int64,
        Prim::Int32,
        Prim::Int16,
        Prim::Int8,
        Prim::Bool,
    ];
    for source in dtypes {
        for target in dtypes {
            let ty = |precision| TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision,
            };
            let mut dag = Dag::new();
            let a = dag.add_node(RiscOp::synth_const(source, 1.0), vec![], ty(source), None);
            let middle = if source == Prim::Bool {
                RiscOp::Copy
            } else {
                RiscOp::Neg
            };
            let b = dag.add_node(middle, vec![a], ty(source), None);
            let c = dag.add_node(
                RiscOp::Cast {
                    new_precision: target,
                },
                vec![b],
                ty(target),
                None,
            );
            dag.add_root(c);
            for hip in [false, true] {
                let verified = verify_ownership(lower_dag_ownership(dag.clone()).unwrap()).unwrap();
                let shared = if hip {
                    let plan = plan_hip_storage(verified).unwrap();
                    plan.slot_for_node(a) == plan.slot_for_node(c)
                } else {
                    let plan = plan_c_storage(verified).unwrap();
                    plan.slot_for_node(a) == plan.slot_for_node(c)
                };
                assert_eq!(
                    shared,
                    source == target,
                    "source={source:?} target={target:?} hip={hip}"
                );
            }
        }
    }
}
