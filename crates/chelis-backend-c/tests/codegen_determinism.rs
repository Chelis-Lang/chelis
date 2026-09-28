//! C codegen emits byte-identical source for a fixed DAG, and entry
//! validation follows ABI slot order, including when labels are not
//! lexically sorted.

mod support;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::codegen;

const INPUT_NAMES: [&str; 8] = [
    "zeta", "alpha", "mu", "delta", "kappa", "beta", "gamma", "epsilon",
];

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

/// A DAG with multiple distinct Load names. The number of inputs is large
/// enough that UnordMap-iteration non-determinism would almost certainly
/// surface across repeated emissions if it were still present.
fn build_multi_input_dag() -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let mut loads = Vec::with_capacity(INPUT_NAMES.len());
    for name in &INPUT_NAMES {
        let id = dag.add_node(
            decl,
            RiscOp::Load {
                name: (*name).into(),
            },
            vec![],
            vec_f32(4),
            None,
        );
        loads.push(id);
    }
    // Chain adds so the DAG is realistic (otherwise the Loads might be
    // optimized away or pruned in some future pass).
    let mut acc = loads[0];
    for next in loads.iter().skip(1) {
        acc = dag.add_node(decl, RiscOp::Add, vec![acc, *next], vec_f32(4), None);
    }
    dag
}

#[test]
fn codegen_is_byte_deterministic_across_repeated_emissions() {
    let dag = build_multi_input_dag();
    let baseline = codegen(&dag, "multi_input").unwrap().c_source;
    // Run many times; every emission constructs fresh collections internally,
    // so any residual hash-iteration non-determinism would show up here.
    for i in 0..32 {
        let again = codegen(&dag, "multi_input").unwrap().c_source;
        assert_eq!(
            baseline, again,
            "C codegen must be byte-deterministic; iteration {i} differs"
        );
    }
}

#[test]
fn input_validation_preamble_follows_assigned_abi_slots() {
    let dag = build_multi_input_dag();
    let src = codegen(&dag, "multi_input").unwrap().c_source;

    // The first two names alone make a lexical reorder differ from ABI order.
    assert!(INPUT_NAMES[0] > INPUT_NAMES[1]);
    let positions: Vec<_> = INPUT_NAMES
        .iter()
        .enumerate()
        .map(|(slot, name)| {
            let diagnostic = format!("input `{name}` at slot {slot} is NULL");
            assert_eq!(
                src.matches(&diagnostic).count(),
                1,
                "each assigned slot must have one NULL diagnostic: {src}"
            );
            src.find(&diagnostic).expect("diagnostic was counted")
        })
        .collect();
    assert!(
        positions.windows(2).all(|pair| pair[0] < pair[1]),
        "input-validation diagnostics must follow assigned ABI slots: {src}"
    );
}
