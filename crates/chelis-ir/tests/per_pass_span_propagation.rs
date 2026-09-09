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
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        Some("dce.a".into()),
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 2.0),
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
        RiscOp::synth_const(scalar_f32().precision, 99.0),
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
    let a = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        None,
    );
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
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        Some("verify.a".into()),
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 2.0),
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
        RiscOp::synth_const(scalar_f32().precision, 3.0),
        vec![],
        scalar_f32(),
        Some("eval.a".into()),
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 4.0),
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
    assert_eq!(result[&sum].to_f64_lossy_vec(), vec![7.0]);

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
        RiscOp::synth_const(scalar_f32().precision, 2.0),
        vec![],
        scalar_f32(),
        Some("lit.a".into()),
    );
    let lit_b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 3.0),
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
        matches!(folded.op, RiscOp::Const { value } if (value.as_f64_lossy() - 5.0).abs() < f64::EPSILON),
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
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        Some("shared".into()),
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 2.0),
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
        RiscOp::synth_const(scalar_f32().precision, 5.0),
        vec![],
        scalar_f32(),
        Some("u.lit".into()),
    );
    let op = dag.add_node(RiscOp::Neg, vec![lit], scalar_f32(), Some("u.neg".into()));
    dag.add_root(op);

    optimize::constant_fold(&mut dag);

    let folded = dag.get(op).expect("op node exists");
    assert!(
        matches!(folded.op, RiscOp::Const { value } if (value.as_f64_lossy() + 5.0).abs() < f64::EPSILON)
    );
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
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        Some("lit.a".into()),
    );
    dag.node_mut(lit_a).unwrap().merged_spans = vec!["lit.a.alias".into()];
    let lit_b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 2.0),
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
    let a = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 2.0),
        vec![],
        scalar_f32(),
        None,
    );
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
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        Some("cse.first".into()),
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 1.0),
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
        RiscOp::synth_const(scalar_f32().precision, 7.0),
        vec![],
        scalar_f32(),
        Some("cse.first".into()),
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 7.0),
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
        RiscOp::synth_const(scalar_f32().precision, 5.0),
        vec![],
        scalar_f32(),
        Some("cse.first".into()),
    );
    dag.node_mut(a).unwrap().merged_spans = vec!["cse.first.alias".into()];
    let b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 5.0),
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
    let a = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        None,
    );
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

/// When a Tier 2 helper decomposes into sub-nodes, each sub-node
/// inherits the decomposed parent's `span_id`. Per
/// spec/design/chelis_span_survival.md §2.3 Tier 2 row.
///
/// In the current lowering, `lower_div` is a degenerate decomposition (one
/// synthesized `RiscOp::Div` node — the cascade was collapsed), so
/// we exercise the rule with `lower_sigmoid` which still decomposes
/// into 4 synthesized sub-nodes (`Neg`, `Exp`, `Add`, `Recip`). The
/// single-node `lower_div` span attribution is locked separately
/// below in `tier2_lower_div_synthesized_node_inherits_parent_span`.
#[test]
fn tier2_sub_nodes_inherit_parent_span() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 0.5),
        vec![],
        scalar_f32(),
        None,
    );
    let result = tier2::lower_sigmoid(&mut dag, x, &scalar_f32(), Some("sigmoid.expr"));

    // The operand const has no span. The sub-nodes are Neg, Exp, Add,
    // and Recip. Every synthesized one should carry span_id =
    // "sigmoid.expr".
    let mut synth_count = 0usize;
    for node in dag.nodes() {
        if matches!(node.op, RiscOp::Const { .. }) {
            // Operand consts (and the synthesized const(1.0) inside
            // lower_sigmoid) — the const(1.0) is also synthesized and
            // should carry the span; check it explicitly.
            if matches!(node.span_id.as_deref(), Some("sigmoid.expr")) {
                synth_count += 1;
            }
            continue;
        }
        synth_count += 1;
        assert_eq!(
            node.span_id.as_deref(),
            Some("sigmoid.expr"),
            "Tier 2 sub-node {:?} ({:?}) should inherit parent span",
            node.id,
            node.op,
        );
    }
    assert!(
        synth_count >= 4,
        "expected at least 4 tier2 sub-nodes carrying parent span, got {synth_count}"
    );
    assert!(matches!(dag.get(result).unwrap().op, RiscOp::Recip));
}

/// `lower_div` is a degenerate Tier 2 lowering: it emits a single
/// synthesized `RiscOp::Div` node. The span-propagation rule still
/// applies to that lone node.
#[test]
fn tier2_lower_div_synthesized_node_inherits_parent_span() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 6.0),
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 3.0),
        vec![],
        scalar_f32(),
        None,
    );
    let result = tier2::lower_div(&mut dag, a, b, &scalar_f32(), Some("div.expr"));

    let node = dag.get(result).unwrap();
    assert!(
        matches!(node.op, RiscOp::Div),
        "lower_div emits a single RiscOp::Div node"
    );
    assert_eq!(node.span_id.as_deref(), Some("div.expr"));
}

/// When the parent op had no source span (e.g. a hand-written
/// non-span-bearing program), Tier 2 sub-nodes carry the canonical
/// `__synthesized_tier2__` marker instead. Per §2.3 Tier 2
/// synthesized-node rule.
#[test]
fn tier2_sub_nodes_use_synthesized_marker_when_parent_has_no_span() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        None,
    );
    // No parent span — sub-nodes should get __synthesized_tier2__.
    let _ = tier2::lower_relu(&mut dag, x, &scalar_f32(), None);

    // The dedicated ReLU identity carries the synthesized marker. The
    // original Const(1) input does not.
    let mut marker_count = 0usize;
    for node in dag.nodes() {
        match node.span_id.as_deref() {
            Some(s) if s == tier2::TIER2_SYNTH_MARKER => {
                marker_count += 1;
            }
            None => {
                // The pre-existing operand Const(1.0). OK.
                assert!(matches!(node.op, RiscOp::Const { value } if value.as_f64_lossy() == 1.0));
            }
            other => panic!(
                "unexpected span_id {other:?} on node {:?} ({:?})",
                node.id, node.op
            ),
        }
    }
    assert!(
        marker_count == 1,
        "expected exactly one ReLU identity carrying __synthesized_tier2__, got {marker_count}"
    );
}

/// Direct subtraction retains its own identity and the parent's span; it
/// must not reconstruct subtraction through synthetic Add/Neg nodes.
#[test]
fn tier2_lower_sub_preserves_direct_identity_and_parent_span() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 5.0),
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 3.0),
        vec![],
        scalar_f32(),
        None,
    );
    let result = tier2::lower_sub(&mut dag, a, b, &scalar_f32(), Some("sub.expr"));
    assert_eq!(dag.len(), 3, "direct sub should add exactly one node");
    let sub = dag.get(result).expect("lowered Sub node");
    assert_eq!(sub.op, RiscOp::Sub);
    assert_eq!(sub.inputs, vec![a, b]);
    assert_eq!(sub.span_id.as_deref(), Some("sub.expr"));
    assert!(
        dag.nodes()
            .iter()
            .all(|node| !matches!(node.op, RiscOp::Add | RiscOp::Neg)),
        "direct Sub lowering must not synthesize Add/Neg nodes"
    );

    for node in dag.nodes() {
        match (&node.op, node.span_id.as_deref()) {
            (RiscOp::Const { .. }, None) => {} // operand consts
            (RiscOp::Sub, Some("sub.expr")) => {}
            (op, span) => panic!(
                "unexpected (op={op:?}, span={span:?}) on node {:?}",
                node.id
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// S3.7 — AD: backward nodes carry __synthesized_grad__ + forward span
// ─────────────────────────────────────────────────────────────────────

/// Mandatory test from spec §3 S3 (AD three-hop). Lower a span-bearing
/// forward program (mul of two span-bearing operands), run AD, walk the
/// resulting DAG. For each backward node, assert
/// `span_id == "__synthesized_grad__"` AND `merged_spans` contains the
/// corresponding forward node's `span_id`. This proves the audit chain
/// (backward → forward → LaTeX source) is reconstructible.
#[test]
fn ad_backward_nodes_carry_grad_marker_and_forward_span() {
    use grad::{GRAD_SYNTH_MARKER, grad_dag};

    // Forward: y = a * b. Both literals + the mul carry distinct spans.
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        scalar_f32(),
        Some("fwd.a".into()),
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        scalar_f32(),
        Some("fwd.b".into()),
    );
    let y = dag.add_node(
        RiscOp::Mul,
        vec![a, b],
        scalar_f32(),
        Some("fwd.mul".into()),
    );
    dag.add_root(y);

    let result = grad_dag(&dag, y, &[a, b]).expect("grad_dag should succeed on scalar mul");

    // Walk the resulting DAG. Forward nodes (Load("a"), Load("b"), Mul)
    // keep their original spans. Every other node is a backward node
    // and must carry GRAD_SYNTH_MARKER as span_id, plus the forward
    // span in merged_spans.
    let forward_spans: BTreeSet<&str> = ["fwd.a", "fwd.b", "fwd.mul"].iter().copied().collect();
    let mut saw_grad_marker = false;
    let mut saw_fwd_mul_in_merged = false;

    for node in result.dag.nodes() {
        match node.span_id.as_deref() {
            Some(s) if s == GRAD_SYNTH_MARKER => {
                saw_grad_marker = true;
                // Every grad-marker node MUST have non-empty merged_spans
                // (per §3 S3 oracle: synthesized markers never the only
                // provenance) — and the merged spans MUST include a
                // forward span.
                assert!(
                    !node.merged_spans.is_empty(),
                    "grad-marker node {:?} has empty merged_spans (audit invariant violated)",
                    node.id
                );
                let has_forward = node
                    .merged_spans
                    .iter()
                    .any(|s| forward_spans.contains(s.as_str()));
                assert!(
                    has_forward,
                    "grad-marker node {:?} merged_spans {:?} contains no forward span",
                    node.id, node.merged_spans
                );
                if node.merged_spans.iter().any(|s| s == "fwd.mul") {
                    saw_fwd_mul_in_merged = true;
                }
            }
            Some(s) if forward_spans.contains(s) => {
                // Forward node — span should be preserved unchanged.
            }
            other => panic!(
                "unexpected span_id {other:?} on node {:?} ({:?})",
                node.id, node.op
            ),
        }
    }
    assert!(saw_grad_marker, "no node carried GRAD_SYNTH_MARKER");
    // The seed (∂y/∂y = 1) and the Mul's adjoints all reference fwd.mul.
    // That's the AD rule: backward nodes attribute to the FORWARD NODE
    // BEING DIFFERENTIATED, not its operands. The operands' spans
    // survive on their forward nodes (Load("a"), Load("b")), so the
    // global audit invariant ("every input span appears as span_id or
    // merged_spans on at least one node") still holds — just not on
    // backward nodes specifically.
    assert!(
        saw_fwd_mul_in_merged,
        "no grad-marker node folded fwd.mul into merged_spans"
    );
    // Confirm the global audit invariant: every forward span survives
    // somewhere in the result DAG.
    let result_spans = dag_spans(&result.dag);
    for fwd in &forward_spans {
        assert!(
            result_spans.contains(*fwd),
            "forward span {fwd} dropped by AD; result spans: {result_spans:?}"
        );
    }
}

/// Forward nodes carry their original span_id and merged_spans through
/// AD unchanged. Per §2.3 AD row: "Forward nodes: clone span_id and
/// merged_spans."
#[test]
fn ad_forward_nodes_preserve_their_spans() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        scalar_f32(),
        Some("fwd.a".into()),
    );
    // Stamp a merged_span on the forward Load to confirm it survives.
    dag.node_mut(a).unwrap().merged_spans = vec!["fwd.a.merged".into()];
    let y = dag.add_node(RiscOp::Neg, vec![a], scalar_f32(), Some("fwd.neg".into()));
    dag.add_root(y);

    let result = grad::grad_dag(&dag, y, &[a]).expect("grad_dag should succeed");
    let spans = dag_spans(&result.dag);
    assert!(
        spans.contains("fwd.a"),
        "forward span fwd.a lost: {spans:?}"
    );
    assert!(
        spans.contains("fwd.a.merged"),
        "forward merged_span fwd.a.merged lost: {spans:?}"
    );
    assert!(
        spans.contains("fwd.neg"),
        "forward span fwd.neg lost: {spans:?}"
    );
}

/// Negative parity: AD on a span-less forward DAG produces backward
/// nodes carrying ONLY GRAD_SYNTH_MARKER (no forward span to merge), so
/// the merged_spans on grad-marker nodes is empty. Note this
/// specifically violates the audit-invariant for synthesized markers
/// that the spec requires — but that invariant only applies when the
/// FORWARD DAG carries spans. With no forward spans, there's nothing
/// to fold; this is the legitimate "unspanned input" case.
#[test]
fn ad_on_unspanned_forward_dag_does_not_fabricate_spans() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        scalar_f32(),
        None,
    );
    let y = dag.add_node(RiscOp::Mul, vec![a, b], scalar_f32(), None);
    dag.add_root(y);

    let result = grad::grad_dag(&dag, y, &[a, b]).expect("grad_dag");
    for node in result.dag.nodes() {
        match node.span_id.as_deref() {
            // Forward nodes inherit None.
            None => assert!(
                node.merged_spans.is_empty(),
                "merged_spans on unspanned forward node should be empty"
            ),
            // Backward nodes still get the marker — but without forward
            // spans, merged_spans is empty (consistent with the
            // "audit-invariant only when input has spans" carve-out).
            Some(s) if s == grad::GRAD_SYNTH_MARKER => {
                assert!(
                    node.merged_spans.is_empty(),
                    "unspanned-forward AD fabricated merged_spans: {:?}",
                    node.merged_spans
                );
            }
            other => panic!(
                "unexpected span_id {other:?} on AD-on-unspanned node {:?}",
                node.id
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// S3 named oracle (after S3.8 lands) — end-to-end span survival
// ─────────────────────────────────────────────────────────────────────

/// S3 named acceptance oracle (per spec §3 S3): lower a span-attributed
/// Deep program through every enabled pass (lowering → DCE → fold →
/// CSE → tier-2 if applicable → fusion → vmap if applicable). Walk the
/// final IR. Assert:
/// (a) every input Deep span appears as `span_id` or in `merged_spans`
///     on at least one final IR node, AND
/// (b) every `__synthesized_*__` marker has a non-empty `merged_spans`
///     (synthesized markers are NEVER the only provenance — they always
///     carry forward-node spans alongside).
///
/// AD is exercised separately in `s3_oracle_with_ad` because grad_dag
/// requires a scalar-float root; a single span-rich program covering
/// every pass at once would be brittle, so we split into two
/// representative oracles.
#[test]
fn s3_oracle_lowering_then_optimization_passes() {
    use chelis_deep::Expr;
    use chelis_ir::lower_program;
    use chelis_types::{check_ir_program, check_linearity};
    use std::collections::BTreeSet;

    // A small but rich program: direct Tier-1 sub identity, constants
    // (fold-eligible if operands match), repeated subexpression
    // (CSE-eligible). Every node carries its own span.
    let source = r#"
        (def {span: "src.def"} y
          (app {type: (t-tensor {} (t-prim {} f32)) span: "src.outer_add"}
               (var {} add)
               (app {type: (t-tensor {} (t-prim {} f32)) span: "src.sub"}
                    (var {} sub)
                    (lit {type: (t-tensor {} (t-prim {} f32)) span: "src.lit_a"} 5.0)
                    (lit {type: (t-tensor {} (t-prim {} f32)) span: "src.lit_b"} 3.0))
               (app {type: (t-tensor {} (t-prim {} f32)) span: "src.inner_add"}
                    (var {} add)
                    (lit {type: (t-tensor {} (t-prim {} f32)) span: "src.lit_c"} 1.0)
                    (lit {type: (t-tensor {} (t-prim {} f32)) span: "src.lit_d"} 2.0))))
    "#;

    fn collect_input_spans(exprs: &[Expr]) -> BTreeSet<String> {
        fn walk(expr: &Expr, acc: &mut BTreeSet<String>) {
            if let Some(s) = expr.span_id() {
                acc.insert(s.to_owned());
            }
            match expr {
                Expr::List(list, _) => list.elements.iter().for_each(|child| walk(child, acc)),
                Expr::Node(node, _) => {
                    node.meta().visit_syntax(&mut |_, value| walk(value, acc));
                    node.children_slice()
                        .iter()
                        .for_each(|child| walk(child, acc));
                }
                Expr::Map(map, _) => map.visit_syntax(&mut |_, value| walk(value, acc)),
                Expr::MetaExpr(meta, _) => {
                    meta.metadata.visit_syntax(&mut |_, value| walk(value, acc));
                    walk(&meta.expr, acc);
                }
                Expr::BareList(elements, _) => elements.iter().for_each(|child| walk(child, acc)),
                Expr::UnknownForm(data) => {
                    data.meta.visit_syntax(&mut |_, value| walk(value, acc));
                    data.children.iter().for_each(|child| walk(child, acc));
                }
                Expr::Atom(_, _) => {}
            }
        }
        let mut out = BTreeSet::new();
        for e in exprs {
            walk(e, &mut out);
        }
        out
    }

    let exprs = chelis_deep::parser::parse_str(source).expect("deep parse");
    let input_spans = collect_input_spans(&exprs);
    assert!(
        input_spans.len() >= 7,
        "fixture must carry many spans; got {input_spans:?}"
    );

    let checked = check_ir_program(&exprs).expect("IR check");
    let checked = chelis_effects::check_program(&checked).expect("effects");
    let checked = check_linearity(&checked).expect("linearity");

    // Stage 1: lowering (which already runs DCE inside lower_program_to_library).
    let mut dag = lower_program(&checked);

    // Stage 2: constant fold + CSE + a redundant DCE (all pure on already-DCE'd input).
    optimize::constant_fold(&mut dag);
    let dag = optimize::common_subexpr_eliminate(&dag);
    let dag = optimize::dead_code_eliminate(&dag);

    // Stage 3: fusion (turn elementwise chains into FusedElem nodes).
    let dag = fuse::fuse(&dag);

    // (a) Audit invariant: every input span appears on at least one
    // final node (as span_id or in merged_spans).
    let final_spans = dag_spans(&dag);
    let missing: Vec<&String> = input_spans.difference(&final_spans).collect();
    assert!(
        missing.is_empty(),
        "S3 oracle: input spans {missing:?} dropped through pipeline. \
         Input: {input_spans:?}; final DAG: {final_spans:?}",
    );

    // (b) Every __synthesized_*__ marker has non-empty merged_spans.
    for node in dag.nodes() {
        if let Some(s) = &node.span_id
            && s.starts_with("__synthesized_")
        {
            assert!(
                !node.merged_spans.is_empty(),
                "S3 oracle: synthesized marker `{s}` on node {:?} has empty merged_spans \
                 (audit invariant violated. Markers must always carry a forward span alongside)",
                node.id
            );
        }
    }
}

/// AD half of the S3 oracle: span-bearing forward program → AD → walk.
/// Same two assertions: (a) audit invariant, (b) every synthesized
/// marker has non-empty merged_spans.
#[test]
fn s3_oracle_with_ad() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        scalar_f32(),
        Some("ad.a".into()),
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        scalar_f32(),
        Some("ad.b".into()),
    );
    let y = dag.add_node(RiscOp::Mul, vec![a, b], scalar_f32(), Some("ad.mul".into()));
    dag.add_root(y);

    let input_spans: BTreeSet<String> = ["ad.a", "ad.b", "ad.mul"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();

    let result = grad::grad_dag(&dag, y, &[a, b]).expect("grad_dag");

    // (a) Audit invariant.
    let final_spans = dag_spans(&result.dag);
    let missing: Vec<&String> = input_spans.difference(&final_spans).collect();
    assert!(
        missing.is_empty(),
        "S3 AD oracle: input spans {missing:?} dropped. \
         Input: {input_spans:?}; final: {final_spans:?}",
    );

    // (b) Every synthesized marker has non-empty merged_spans.
    for node in result.dag.nodes() {
        if let Some(s) = &node.span_id
            && s.starts_with("__synthesized_")
        {
            assert!(
                !node.merged_spans.is_empty(),
                "S3 AD oracle: synthesized marker `{s}` on node {:?} has empty merged_spans",
                node.id
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// S3.8 — Fusion: FusedElem aggregates contributors' spans
// ─────────────────────────────────────────────────────────────────────

/// Build a chain of 3 fusible elementwise ops with distinct spans, run
/// fusion, assert FusedElem `span_id == first.span_id` and
/// `merged_spans` lex-sorted = sort_dedup(rest contributors' spans).
/// Per spec §2.3 Fusion row.
#[test]
fn fusion_aggregates_contributors_spans() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 1.0),
        vec![],
        vec_f32(4),
        Some("fuse.a".into()),
    );
    let b = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 2.0),
        vec![],
        vec_f32(4),
        Some("fuse.b".into()),
    );
    // Three ops in chain: Add → Neg → Exp, each with its own span.
    let add = dag.add_node(
        RiscOp::Add,
        vec![a, b],
        vec_f32(4),
        Some("fuse.first".into()),
    );
    let neg = dag.add_node(RiscOp::Neg, vec![add], vec_f32(4), Some("fuse.mid".into()));
    let exp = dag.add_node(RiscOp::Exp, vec![neg], vec_f32(4), Some("fuse.last".into()));
    dag.add_root(exp);

    let fused = fuse::fuse(&dag);

    // Find the FusedElem.
    let fused_elem = fused
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::FusedElem { .. }))
        .expect("expected a FusedElem");
    assert_eq!(
        fused_elem.span_id.as_deref(),
        Some("fuse.first"),
        "FusedElem must inherit FIRST contributor's span_id"
    );
    assert_eq!(
        fused_elem.merged_spans,
        vec!["fuse.last".to_string(), "fuse.mid".to_string()],
        "FusedElem must aggregate the rest of the chain's spans, lex-sorted"
    );
}

/// Each contributor's pre-existing merged_spans must also flow into the
/// FusedElem (transitive). Locks the rule's "∪ each contributor's
/// pre-existing merged_spans" half.
#[test]
fn fusion_propagates_contributor_merged_spans() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 1.0),
        vec![],
        vec_f32(4),
        Some("fuse.a".into()),
    );
    let b = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 2.0),
        vec![],
        vec_f32(4),
        Some("fuse.b".into()),
    );
    let add = dag.add_node(
        RiscOp::Add,
        vec![a, b],
        vec_f32(4),
        Some("fuse.first".into()),
    );
    // Stamp a merged_span on the first contributor (carries through
    // verbatim).
    dag.node_mut(add).unwrap().merged_spans = vec!["fuse.first.alias".into()];
    let neg = dag.add_node(RiscOp::Neg, vec![add], vec_f32(4), Some("fuse.mid".into()));
    // Stamp a merged_span on a non-first contributor (folds in via
    // append_spans_to_node).
    dag.node_mut(neg).unwrap().merged_spans = vec!["fuse.mid.alias".into()];
    dag.add_root(neg);

    let fused = fuse::fuse(&dag);
    let fused_elem = fused
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::FusedElem { .. }))
        .expect("expected a FusedElem");
    assert_eq!(fused_elem.span_id.as_deref(), Some("fuse.first"));
    assert_eq!(
        fused_elem.merged_spans,
        vec![
            "fuse.first.alias".to_string(),
            "fuse.mid".to_string(),
            "fuse.mid.alias".to_string(),
        ],
    );
}

/// Non-chain nodes (consumers and inputs to a fused chain) are pure
/// copies — span_id + merged_spans verbatim. Locks the second add_node
/// path in `rebuild_with_fusion`.
#[test]
fn fusion_preserves_unfused_node_spans() {
    let mut dag = Dag::new();
    // Build a multi-consumer node so it can't be fused into the chain.
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        Some("unfused.load".into()),
    );
    dag.node_mut(a).unwrap().merged_spans = vec!["unfused.load.alias".into()];
    // Two consumers prevent fusion through `a`.
    let _consumer1 = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4), Some("unfused.c1".into()));
    let _consumer2 = dag.add_node(RiscOp::Exp, vec![a], vec_f32(4), Some("unfused.c2".into()));
    dag.add_root(a);

    let fused = fuse::fuse(&dag);
    let load_node = fused
        .nodes()
        .iter()
        .find(|n| matches!(n.op, RiscOp::Load { .. }))
        .expect("Load survives");
    assert_eq!(load_node.span_id.as_deref(), Some("unfused.load"));
    assert_eq!(
        load_node.merged_spans,
        vec!["unfused.load.alias".to_string()]
    );
}

/// Negative parity: fusion on unspanned input fabricates nothing.
#[test]
fn fusion_does_not_fabricate_spans() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 1.0),
        vec![],
        vec_f32(4),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 2.0),
        vec![],
        vec_f32(4),
        None,
    );
    let add = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4), None);
    let neg = dag.add_node(RiscOp::Neg, vec![add], vec_f32(4), None);
    dag.add_root(neg);

    let fused = fuse::fuse(&dag);
    for node in fused.nodes() {
        assert_eq!(
            node.span_id, None,
            "fusion fabricated span_id on node {:?}",
            node.id
        );
        assert!(
            node.merged_spans.is_empty(),
            "fusion fabricated merged_spans on node {:?}",
            node.id
        );
    }
}

/// DCE remap variant (used by Phase F library carrier) must apply the
/// same pure-copy rule. Locking it explicitly so an alternate code path
/// can't drift from the headline `dead_code_eliminate`.
#[test]
fn dce_with_remap_preserves_spans() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 1.0),
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
