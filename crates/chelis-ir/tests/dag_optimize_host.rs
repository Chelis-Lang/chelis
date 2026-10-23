use chelis_dag_core::dag::*;
use chelis_dag_core::optimize::*;

fn scalar_f32() -> TensorType {
    TensorType::scalar_f32()
}

#[test]
fn cse_keeps_each_key_operation_and_its_draw() {
    // Two `key_from_seed` nodes over one seed produce equal bits, but
    // merging them would hand one key to two draws, which the key rules
    // reject (spec/10 section 3.2, V2): CSE keeps both key operations and
    // both draws, and the two draws, as pure functions of equal keys,
    // produce equal values.
    use chelis_numeric_core::prim::Prim;
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let template = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        scalar_f32(),
        None,
    );
    let seed = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::Int64, 42.0),
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        None,
    );
    let low = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::F32, 2.0),
        vec![],
        scalar_f32(),
        None,
    );
    let high = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::F32, 5.0),
        vec![],
        scalar_f32(),
        None,
    );
    for _ in 0..2 {
        let key = dag.add_node(
            decl,
            RiscOp::KeyFromSeed,
            vec![seed],
            TensorType {
                dims: vec![],
                precision: Prim::Key,
            },
            None,
        );
        let draw = dag.add_node(
            decl,
            RiscOp::UniformLike,
            vec![template, low, high, key],
            scalar_f32(),
            None,
        );
        dag.add_root(draw);
    }
    let optimized = common_subexpr_eliminate(&dag);
    assert_eq!(
        chelis_dag_core::verify::verify(&optimized),
        Vec::<String>::new()
    );
    assert_eq!(
        optimized
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::KeyFromSeed))
            .count(),
        2,
        "equal key operations must not merge"
    );
    assert_eq!(optimized.roots().len(), 2);
    assert_ne!(optimized.roots()[0], optimized.roots()[1]);
    let values =
        chelis_ir::eval::eval_tensor_roots_exact(&optimized, optimized.roots(), |_| None).unwrap();
    assert_eq!(values[&optimized.roots()[0]], values[&optimized.roots()[1]]);
}
