//! Regression: C codegen must be byte-deterministic across runs.
//!
//! Per `spec/upstream-bugs/host-emit-hashmap-iteration-nondeterminism.md`,
//! `chelis_backend_c::emit::CEmitter::emit_input_shape_preamble` previously
//! iterated a `HashMap<String, TensorType>` of input parameters, whose
//! iteration order was non-deterministic across process runs, and across
//! compilations within a single run because each emitter call constructed a
//! freshly seeded map. Entry validation now follows assigned ABI input slots.
//!
//! The user-visible symptom was that the input-validation block (NULL
//! checks, ndim checks, fixed-axis-size checks, and symbolic-dim binding
//! lines) emitted in different orders for different invocations of
//! `chelis build` against the same `.dp` source, breaking byte-equal
//! reproducible builds. A stable ABI slot order preserves determinism and
//! decides which malformed input fails first.
//!
//! The two assertions below lock the invariant:
//!   1. **Cross-process determinism.** Building the same DAG in many fresh
//!      `CEmitter` instances within one test run produces byte-identical C.
//!      A hash map seeded per instance, so multiple maps within one process
//!      exercised the same non-determinism a fresh `chelis build` would have.
//!      The assertion still holds against an ordered store; it fails if
//!      unstable iteration returns.
//!   2. **ABI slot order.** The input-validation block follows assigned input
//!      slots even when their labels are deliberately not lex-sorted.

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
