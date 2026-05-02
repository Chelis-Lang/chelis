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

// ─────────────────────────────────────────────────────────────────────
// S3.4 — Constant fold: op span survives, operand spans merge in
// ─────────────────────────────────────────────────────────────────────

/// Mandatory test from spec §3 S3 (constant-fold three-source). Fold an
/// expression where the operation and its operands carry distinct span
/// IDs. Assert the folded result carries `span_id = "<op-span>"` and
/// `merged_spans` contains every operand span (lex-sorted, deduped).
#[test]
fn constant_fold_inherits_op_span_and_merges_operands() {
    let mut dag = Dag::new();
    // 2 + 3 with three distinct spans on (literal, literal, op).
    let lit_a = dag.add_node(
        RiscOp::Const { value: 2.0 },
        vec![],
        scalar_f32(),
        Some("lit.a".into()),
    );
    let lit_b = dag.add_node(
        RiscOp::Const { value: 3.0 },
        vec![],
        scalar_f32(),
        Some("lit.b".into()),
    );
    let op = dag.add_node(
        RiscOp::Add,
        vec![lit_a, lit_b],
        scalar_f32(),
        Some("op.plus".into()),
    );
    dag.add_root(op);

    optimize::constant_fold(&mut dag);

    let folded = dag.get(op).expect("op node exists");
    assert!(
        matches!(folded.op, RiscOp::Const { value } if (value - 5.0).abs() < f64::EPSILON),
        "fold should have produced Const(5.0); got {:?}",
        folded.op
    );
    assert_eq!(
        folded.span_id.as_deref(),
        Some("op.plus"),
        "folded result must inherit the operation node's span_id"
    );
    assert_eq!(
        folded.merged_spans,
        vec!["lit.a".to_string(), "lit.b".to_string()],
        "operand spans must merge into folded result lex-sorted"
    );
}

/// Negative parity: when operands share the operation's span (or the
/// operation has no span), the operand spans STILL flow into
/// `merged_spans` only when they're distinct from the canonical and not
/// already present. This locks the dedup semantics.
#[test]
fn constant_fold_dedups_operand_span_matching_op_span() {
    let mut dag = Dag::new();
    // Both operands carry the SAME span as the op. After fold,
    // merged_spans should be empty (canonical-no-op dedup).
    let a = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        scalar_f32(),
        Some("shared".into()),
    );
    let b = dag.add_node(
        RiscOp::Const { value: 2.0 },
        vec![],
        scalar_f32(),
        Some("shared".into()),
    );
    let op = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), Some("shared".into()));
    dag.add_root(op);

    optimize::constant_fold(&mut dag);

    let folded = dag.get(op).expect("op node exists");
    assert_eq!(folded.span_id.as_deref(), Some("shared"));
    assert!(
        folded.merged_spans.is_empty(),
        "expected dedup against canonical span; got {:?}",
        folded.merged_spans
    );
}

/// Unary fold variant: neg of a span-bearing const. The operand's span
/// flows into merged_spans even when the op carries its own.
#[test]
fn constant_fold_unary_merges_operand_span() {
    let mut dag = Dag::new();
    let lit = dag.add_node(
        RiscOp::Const { value: 5.0 },
        vec![],
        scalar_f32(),
        Some("u.lit".into()),
    );
    let op = dag.add_node(RiscOp::Neg, vec![lit], scalar_f32(), Some("u.neg".into()));
    dag.add_root(op);

    optimize::constant_fold(&mut dag);

    let folded = dag.get(op).expect("op node exists");
    assert!(matches!(folded.op, RiscOp::Const { value } if (value + 5.0).abs() < f64::EPSILON));
    assert_eq!(folded.span_id.as_deref(), Some("u.neg"));
    assert_eq!(folded.merged_spans, vec!["u.lit".to_string()]);
}

/// Operand's pre-existing merged_spans must also flow onto the folded
/// node (transitive — important when fold runs after another pass that
/// already populated merged_spans).
#[test]
fn constant_fold_propagates_operand_merged_spans() {
    let mut dag = Dag::new();
    let lit_a = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        scalar_f32(),
        Some("lit.a".into()),
    );
    dag.node_mut(lit_a).unwrap().merged_spans = vec!["lit.a.alias".into()];
    let lit_b = dag.add_node(
        RiscOp::Const { value: 2.0 },
        vec![],
        scalar_f32(),
        Some("lit.b".into()),
    );
    let op = dag.add_node(
        RiscOp::Add,
        vec![lit_a, lit_b],
        scalar_f32(),
        Some("op.plus".into()),
    );
    dag.add_root(op);

    optimize::constant_fold(&mut dag);

    let folded = dag.get(op).expect("op node exists");
    assert_eq!(folded.span_id.as_deref(), Some("op.plus"));
    assert_eq!(
        folded.merged_spans,
        vec![
            "lit.a".to_string(),
            "lit.a.alias".to_string(),
            "lit.b".to_string()
        ],
    );
}

/// Negative parity: fold on unspanned input fabricates nothing.
#[test]
fn constant_fold_does_not_fabricate_spans() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
    let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32(), None);
    let op = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
    dag.add_root(op);

    optimize::constant_fold(&mut dag);

    let folded = dag.get(op).expect("op node exists");
    assert_eq!(folded.span_id, None);
    assert!(folded.merged_spans.is_empty());
}

// ─────────────────────────────────────────────────────────────────────
// S3.5 — CSE: survivor keeps span_id, duplicate's provenance merges in
// ─────────────────────────────────────────────────────────────────────

/// Two structurally-identical Const nodes with distinct spans collapse
/// onto one survivor under CSE. The survivor keeps the FIRST node's
/// `span_id`; the dropped duplicate's `span_id` lands in the survivor's
/// `merged_spans`. Per §2.3 CSE row.
#[test]
fn cse_merges_duplicate_span_into_survivor() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        scalar_f32(),
        Some("cse.first".into()),
    );
    let b = dag.add_node(
        RiscOp::Const { value: 1.0 },
        vec![],
        scalar_f32(),
        Some("cse.dup".into()),
    );
    let sum = dag.add_node(
        RiscOp::Add,
        vec![a, b],
        scalar_f32(),
        Some("cse.add".into()),
    );
    dag.add_root(sum);

    let new_dag = optimize::common_subexpr_eliminate(&dag);
    // The two consts collapse: 2 nodes (1 Const + 1 Add).
    assert_eq!(new_dag.len(), 2, "CSE should collapse identical consts");

    let const_node = new_dag
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::Const { .. }))
        .expect("Const survives");
    assert_eq!(
        const_node.span_id.as_deref(),
        Some("cse.first"),
        "survivor must keep first-seen span_id"
    );
    assert_eq!(
        const_node.merged_spans,
        vec!["cse.dup".to_string()],
        "duplicate's span must land in survivor's merged_spans"
    );

    // The Add node also keeps its own span unchanged.
    let add_node = new_dag
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::Add))
        .expect("Add survives");
    assert_eq!(add_node.span_id.as_deref(), Some("cse.add"));
}

/// Transitive: a duplicate that already carries pre-existing
/// `merged_spans` must fold all of them onto the survivor (not just the
/// canonical span).
#[test]
fn cse_propagates_duplicate_merged_spans() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 7.0 },
        vec![],
        scalar_f32(),
        Some("cse.first".into()),
    );
    let b = dag.add_node(
        RiscOp::Const { value: 7.0 },
        vec![],
        scalar_f32(),
        Some("cse.dup".into()),
    );
    // Stamp pre-existing merged_spans on the duplicate.
    dag.node_mut(b).unwrap().merged_spans =
        vec!["cse.dup.alias.1".into(), "cse.dup.alias.2".into()];
    let neg = dag.add_node(RiscOp::Neg, vec![a], scalar_f32(), None);
    let neg2 = dag.add_node(RiscOp::Neg, vec![b], scalar_f32(), None);
    dag.add_root(neg);
    dag.add_root(neg2);

    let new_dag = optimize::common_subexpr_eliminate(&dag);

    let const_node = new_dag
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::Const { .. }))
        .expect("Const survives");
    assert_eq!(const_node.span_id.as_deref(), Some("cse.first"));
    assert_eq!(
        const_node.merged_spans,
        vec![
            "cse.dup".to_string(),
            "cse.dup.alias.1".to_string(),
            "cse.dup.alias.2".to_string(),
        ],
        "all of duplicate's provenance (canonical + merged) must fold lex-sorted onto survivor"
    );
}

/// Survivor's pre-existing merged_spans are preserved through CSE
/// (i.e. CSE doesn't drop the survivor's own merged provenance when
/// merging in a duplicate).
#[test]
fn cse_preserves_survivor_merged_spans() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 5.0 },
        vec![],
        scalar_f32(),
        Some("cse.first".into()),
    );
    dag.node_mut(a).unwrap().merged_spans = vec!["cse.first.alias".into()];
    let b = dag.add_node(
        RiscOp::Const { value: 5.0 },
        vec![],
        scalar_f32(),
        Some("cse.dup".into()),
    );
    let sum = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
    dag.add_root(sum);

    let new_dag = optimize::common_subexpr_eliminate(&dag);
    let const_node = new_dag
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::Const { .. }))
        .expect("Const survives");
    assert_eq!(
        const_node.merged_spans,
        vec!["cse.dup".to_string(), "cse.first.alias".to_string()],
    );
}

/// Negative parity: CSE on unspanned input fabricates nothing.
#[test]
fn cse_does_not_fabricate_spans() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
    let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
    let sum = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
    dag.add_root(sum);

    let new_dag = optimize::common_subexpr_eliminate(&dag);
    for node in new_dag.nodes() {
        assert_eq!(node.span_id, None, "CSE fabricated span_id");
        assert!(node.merged_spans.is_empty(), "CSE fabricated merged_spans");
    }
}

// ─────────────────────────────────────────────────────────────────────
// S3.6 — Tier 2 decomposition: sub-nodes inherit parent span (or marker)
// ─────────────────────────────────────────────────────────────────────

/// When a Tier 2 helper (e.g. `lower_div`) decomposes into sub-nodes,
/// each sub-node inherits the decomposed parent's `span_id`. Per
/// spec/design/chelis_span_survival.md §2.3 Tier 2 row.
#[test]
fn tier2_sub_nodes_inherit_parent_span() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 6.0 }, vec![], scalar_f32(), None);
    let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
    // Decompose div(a, b) with parent span "div.expr". Every synthesized
    // sub-node (Log, Neg, Exp, Mul) should carry span_id="div.expr".
    let result = tier2::lower_div(&mut dag, a, b, &scalar_f32(), Some("div.expr"));

    // The two operand consts (a, b) have no span (None). The sub-nodes
    // are nodes 2..=5: Log(b), Neg(log_b), Exp(neg_log), Mul(a, recip).
    let mut synth_count = 0usize;
    for node in dag.nodes() {
        if matches!(node.op, RiscOp::Const { .. }) {
            // Operand consts; not synthesized.
            continue;
        }
        synth_count += 1;
        assert_eq!(
            node.span_id.as_deref(),
            Some("div.expr"),
            "Tier 2 sub-node {:?} ({:?}) should inherit parent span",
            node.id,
            node.op,
        );
    }
    assert!(
        synth_count >= 4,
        "expected at least 4 tier2 sub-nodes, got {synth_count}"
    );
    assert!(matches!(dag.get(result).unwrap().op, RiscOp::Mul));
}

/// When the parent op had no source span (e.g. a hand-written
/// non-span-bearing program), Tier 2 sub-nodes carry the canonical
/// `__synthesized_tier2__` marker instead. Per §2.3 Tier 2
/// synthesized-node rule.
#[test]
fn tier2_sub_nodes_use_synthesized_marker_when_parent_has_no_span() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32(), None);
    // No parent span — sub-nodes should get __synthesized_tier2__.
    let _ = tier2::lower_relu(&mut dag, x, &scalar_f32(), None);

    // The relu decomposes into Const(0) + MaxElem. Both should carry the
    // marker. The original Const(1) input does NOT.
    let mut marker_count = 0usize;
    for node in dag.nodes() {
        match node.span_id.as_deref() {
            Some(s) if s == tier2::TIER2_SYNTH_MARKER => {
                marker_count += 1;
            }
            None => {
                // The pre-existing operand Const(1.0). OK.
                assert!(matches!(node.op, RiscOp::Const { value } if value == 1.0));
            }
            other => panic!(
                "unexpected span_id {other:?} on node {:?} ({:?})",
                node.id, node.op
            ),
        }
    }
    assert!(
        marker_count >= 2,
        "expected at least 2 sub-nodes carrying __synthesized_tier2__, got {marker_count}"
    );
}

/// Larger decomposition (sub) — every emitted sub-node carries the
/// parent's span. Locks the rule across multiple Tier 2 helpers.
#[test]
fn tier2_lower_sub_inherits_parent_span() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32(), None);
    let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
    // sub(a,b) = neg(b) + add(a, neg_b). Two synthesized nodes.
    let _ = tier2::lower_sub(&mut dag, a, b, &scalar_f32(), Some("sub.expr"));
    for node in dag.nodes() {
        match (&node.op, node.span_id.as_deref()) {
            (RiscOp::Const { .. }, None) => {} // operand consts
            (RiscOp::Neg, Some("sub.expr")) | (RiscOp::Add, Some("sub.expr")) => {}
            (op, span) => panic!(
                "unexpected (op={op:?}, span={span:?}) on node {:?}",
                node.id
            ),
        }
    }
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
