//! S3 per-pass span-propagation tests + the S3 named oracle.
//!
//! Each transformation pass listed in `spec/design/chelis_span_survival.md`
//! §2.3 has a per-pass test here that locks the row's rule. The final
//! `s3_end_to_end_oracle_*` tests are the named acceptance oracle for
//! phase S3 (see §3 S3): every input Deep span appears as `span_id` or
//! `merged_spans` on at least one final IR node, AND every
//! `__synthesized_*__` marker has non-empty `merged_spans` (synthesized
//! markers are never the only provenance).
//!
//! Style mirrors `span_threading_through_lowering.rs`: positive +
//! negative parity per CLAUDE.md, no implementation-detail leakage.

// File grows commit-by-commit through S3.1–S3.8 + the oracle. Allow the
// "imported but not yet used" warnings rather than re-shuffling imports
// per commit — every name listed here has a callsite by the end of S3.8.
#![allow(dead_code, unused_imports)]

use std::collections::BTreeSet;

use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::{fuse, grad, optimize, tier2, vmap};
use chelis_types::types::Prim;

fn scalar_f32() -> TensorType {
    TensorType::scalar_f32()
}

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

/// Collect every span on a DAG (canonical span_id ∪ merged_spans).
fn dag_spans(dag: &Dag) -> BTreeSet<String> {
    let mut acc = BTreeSet::new();
    for node in dag.nodes() {
        if let Some(s) = &node.span_id {
            acc.insert(s.clone());
        }
        for s in &node.merged_spans {
            acc.insert(s.clone());
        }
    }
    acc
}

// ─────────────────────────────────────────────────────────────────────
// S3.1 — Vmap: clone span_id + merged_spans unchanged
// ─────────────────────────────────────────────────────────────────────

/// Vmap on a span-bearing DAG must preserve every span (canonical and
/// merged). Spec §2.3 vmap row: "Clone `span_id` and `merged_spans`
/// unchanged."
#[test]
fn vmap_preserves_span_id_and_merged_spans() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(3),
        Some("vmap.x_load".into()),
    );
    let y = dag.add_node(RiscOp::Neg, vec![x], vec_f32(3), Some("vmap.neg".into()));
    // Stamp a merged_span onto y to confirm it propagates too.
    dag.node_mut(y).unwrap().merged_spans = vec!["vmap.neg.merged".into()];
    dag.add_root(y);

    let vmapped = vmap::vectorize_axis0(&dag, DimInfo::Lit(2)).expect("vmap should succeed");
    let spans = dag_spans(&vmapped);
    assert!(
        spans.contains("vmap.x_load"),
        "vmap dropped canonical span on Load: {spans:?}"
    );
    assert!(
        spans.contains("vmap.neg"),
        "vmap dropped canonical span on Neg: {spans:?}"
    );
    assert!(
        spans.contains("vmap.neg.merged"),
        "vmap dropped a merged_span: {spans:?}"
    );

    // Per-node parity: each node in the vmapped DAG should have the same
    // canonical span as its source, and the same merged_spans vector.
    for (orig, vmapped_node) in dag.nodes().iter().zip(vmapped.nodes().iter()) {
        assert_eq!(
            orig.span_id, vmapped_node.span_id,
            "vmap altered span_id on node {:?}",
            orig.id
        );
        assert_eq!(
            orig.merged_spans, vmapped_node.merged_spans,
            "vmap altered merged_spans on node {:?}",
            orig.id
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// S3.2 — DCE / remap: pure copy of span_id + merged_spans
// ─────────────────────────────────────────────────────────────────────

/// DCE preserves span_id and merged_spans on every surviving node. Per
/// spec/design/chelis_span_survival.md §2.3 DCE/remap row: "Pure copy;
/// clone `span_id` and `merged_spans` to the remapped node." The S2
/// cleanup already implemented this (optimize.rs:120-144); this test
/// locks that the invariant holds and that the pre-pass grep wasn't
/// missing a pure-copy site.
#[test]
fn dce_preserves_span_id_and_merged_spans() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        scalar_f32(),
        Some("dce.a".into()),
    );
    let b = dag.add_node(
        RiscOp::Const { value: 2.0 },
        vec![],
        scalar_f32(),
        Some("dce.b".into()),
    );
    let live = dag.add_node(
        RiscOp::Add,
        vec![a, b],
        scalar_f32(),
        Some("dce.live".into()),
    );
    // Stamp a merged_span on the live node to confirm DCE preserves it.
    dag.node_mut(live).unwrap().merged_spans = vec!["dce.merged".into()];
    // Add a dead node with its own span — it should be dropped entirely.
    let _dead = dag.add_node(
        RiscOp::Const { value: 99.0 },
        vec![],
        scalar_f32(),
        Some("dce.dead".into()),
    );
    dag.add_root(live);

    let new_dag = optimize::dead_code_eliminate(&dag);

    // Live spans survive.
    let spans = dag_spans(&new_dag);
    assert!(spans.contains("dce.a"), "DCE dropped live span: {spans:?}");
    assert!(spans.contains("dce.b"), "DCE dropped live span: {spans:?}");
    assert!(
        spans.contains("dce.live"),
        "DCE dropped live span: {spans:?}"
    );
    assert!(
        spans.contains("dce.merged"),
        "DCE dropped merged_spans: {spans:?}"
    );
    // Dead node's span is correctly gone.
    assert!(
        !spans.contains("dce.dead"),
        "DCE preserved a dead-only span: {spans:?}"
    );

    // The surviving Add node should still have span_id="dce.live" and
    // merged_spans=["dce.merged"].
    let surviving_add = new_dag
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::Add))
        .expect("Add survives");
    assert_eq!(surviving_add.span_id.as_deref(), Some("dce.live"));
    assert_eq!(surviving_add.merged_spans, vec!["dce.merged"]);
}

/// Negative parity: DCE on unspanned input fabricates nothing.
#[test]
fn dce_does_not_fabricate_spans() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
    let live = dag.add_node(RiscOp::Neg, vec![a], scalar_f32(), None);
    dag.add_root(live);

    let new_dag = optimize::dead_code_eliminate(&dag);
    for node in new_dag.nodes() {
        assert_eq!(node.span_id, None, "DCE fabricated span_id");
        assert!(
            node.merged_spans.is_empty(),
            "DCE fabricated merged_spans: {:?}",
            node.merged_spans,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// S3.3 — Verify / Eval: read-only consumers, no propagation
// ─────────────────────────────────────────────────────────────────────

/// `verify::verify` takes `&Dag` (immutable borrow). Calling it on a
/// span-bearing DAG must leave every span_id and merged_spans value
/// byte-identical. Per spec/design/chelis_span_survival.md §2.3
/// Verify/Eval row: "Read-only; no propagation."
#[test]
fn verify_does_not_mutate_spans() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        scalar_f32(),
        Some("verify.a".into()),
    );
    let b = dag.add_node(
        RiscOp::Const { value: 2.0 },
        vec![],
        scalar_f32(),
        Some("verify.b".into()),
    );
    let sum = dag.add_node(
        RiscOp::Add,
        vec![a, b],
        scalar_f32(),
        Some("verify.sum".into()),
    );
    dag.node_mut(sum).unwrap().merged_spans =
        vec!["verify.merged.x".into(), "verify.merged.y".into()];
    dag.add_root(sum);

    let pre_spans: Vec<(Option<String>, Vec<String>)> = dag
        .nodes()
        .iter()
        .map(|n| (n.span_id.clone(), n.merged_spans.clone()))
        .collect();

    let errors = chelis_ir::verify::verify(&dag);
    assert!(errors.is_empty(), "verify reported errors: {errors:?}");

    let post_spans: Vec<(Option<String>, Vec<String>)> = dag
        .nodes()
        .iter()
        .map(|n| (n.span_id.clone(), n.merged_spans.clone()))
        .collect();
    assert_eq!(
        pre_spans, post_spans,
        "verify mutated DAG spans (read-only contract violated)"
    );
}

/// `eval_tensor_*` likewise takes `&Dag`. Locking the same invariant.
#[test]
fn eval_does_not_mutate_spans() {
    use chelis_ir::eval::eval_tensor_roots_with;
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 3.0 },
        vec![],
        scalar_f32(),
        Some("eval.a".into()),
    );
    let b = dag.add_node(
        RiscOp::Const { value: 4.0 },
        vec![],
        scalar_f32(),
        Some("eval.b".into()),
    );
    let sum = dag.add_node(
        RiscOp::Add,
        vec![a, b],
        scalar_f32(),
        Some("eval.sum".into()),
    );
    dag.node_mut(sum).unwrap().merged_spans = vec!["eval.merged".into()];
    dag.add_root(sum);

    let pre_spans: Vec<(Option<String>, Vec<String>)> = dag
        .nodes()
        .iter()
        .map(|n| (n.span_id.clone(), n.merged_spans.clone()))
        .collect();

    let result = eval_tensor_roots_with(&dag, &[sum], |_| None).expect("eval scalar");
    assert_eq!(result[&sum].data, vec![7.0]);

    let post_spans: Vec<(Option<String>, Vec<String>)> = dag
        .nodes()
        .iter()
        .map(|n| (n.span_id.clone(), n.merged_spans.clone()))
        .collect();
    assert_eq!(
        pre_spans, post_spans,
        "eval mutated DAG spans (read-only contract violated)"
    );
}

/// DCE remap variant (used by Phase F library carrier) must apply the
/// same pure-copy rule. Locking it explicitly so an alternate code path
/// can't drift from the headline `dead_code_eliminate`.
#[test]
fn dce_with_remap_preserves_spans() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        scalar_f32(),
        Some("dcer.a".into()),
    );
    let live = dag.add_node(RiscOp::Neg, vec![a], scalar_f32(), Some("dcer.live".into()));
    dag.node_mut(live).unwrap().merged_spans = vec!["dcer.merged".into()];
    dag.add_root(live);

    let (new_dag, _remap) = optimize::dead_code_eliminate_with_remap(&dag);
    let spans = dag_spans(&new_dag);
    assert!(spans.contains("dcer.a"));
    assert!(spans.contains("dcer.live"));
    assert!(spans.contains("dcer.merged"));
}

/// Negative parity: vmap on an unspanned DAG produces no spans (no
/// fabrication).
#[test]
fn vmap_does_not_fabricate_spans() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(3), None);
    let y = dag.add_node(RiscOp::Neg, vec![x], vec_f32(3), None);
    dag.add_root(y);

    let vmapped = vmap::vectorize_axis0(&dag, DimInfo::Lit(2)).expect("vmap should succeed");
    for node in vmapped.nodes() {
        assert_eq!(
            node.span_id, None,
            "vmap fabricated span_id on node {:?}",
            node.id
        );
        assert!(
            node.merged_spans.is_empty(),
            "vmap fabricated merged_spans on node {:?}: {:?}",
            node.id,
            node.merged_spans
        );
    }
}
