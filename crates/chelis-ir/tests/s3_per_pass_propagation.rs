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
