//! Regression: C codegen must be byte-deterministic across runs.
//!
//! Per `spec/upstream-bugs/host-emit-hashmap-iteration-nondeterminism.md`,
//! `chelis_backend_c::emit::CEmitter::emit_input_shape_preamble` previously
//! iterated a `HashMap<String, TensorType>` of input parameters, whose
//! iteration order was non-deterministic across process runs, and across
//! compilations within a single run because each emitter call constructed a
//! freshly seeded map. That store is a `UnordMap` today and its order is the
//! key's, but the regression this file locks is the original one.
//!
//! The user-visible symptom was that the input-validation block (NULL
//! checks, ndim checks, fixed-axis-size checks, and symbolic-dim binding
//! lines) emitted in different orders for different invocations of
//! `chelis build` against the same `.dp` source, breaking byte-equal
//! reproducible builds. The fix sorts input labels lex before iteration.
//!
//! The two assertions below lock the invariant:
//!   1. **Cross-process determinism.** Building the same DAG in many fresh
//!      `CEmitter` instances within one test run produces byte-identical C.
//!      A hash map seeded per instance, so multiple maps within one process
//!      exercised the same non-determinism a fresh `chelis build` would have.
//!      The assertion still holds against an ordered store; it would now fail
//!      only if something reintroduced an order that is not the key's.
//!   2. **Lex-sorted preamble.** The input-validation block lists labels
//!      in lex order — a stable, observer-visible ordering.

mod support;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::codegen;

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
    let names = [
        "zeta", "alpha", "mu", "delta", "kappa", "beta", "gamma", "epsilon",
    ];
    let mut loads = Vec::with_capacity(names.len());
    for name in &names {
        let id = dag.add_node(
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
        acc = dag.add_node(RiscOp::Add, vec![acc, *next], vec_f32(4), None);
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
fn input_validation_preamble_emits_labels_in_lex_sorted_order() {
    let dag = build_multi_input_dag();
    let src = codegen(&dag, "multi_input").unwrap().c_source;

    // Pull out the order in which input labels appear in the
    // input-validation block. Each label `L` shows up first as an
    // `inputs[N] == NULL` check whose error string contains `` input `L` ``.
    let mut order = Vec::new();
    for line in src.lines() {
        if let Some(open) = line.find("input `") {
            let rest = &line[open + "input `".len()..];
            if let Some(close) = rest.find('`') {
                let label = rest[..close].to_string();
                if !order.contains(&label) {
                    order.push(label);
                }
            }
        }
    }

    let mut expected: Vec<String> = [
        "zeta", "alpha", "mu", "delta", "kappa", "beta", "gamma", "epsilon",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    expected.sort();

    assert_eq!(
        order, expected,
        "input-validation preamble must list labels in lex-sorted order; got:\n{src}"
    );
}
