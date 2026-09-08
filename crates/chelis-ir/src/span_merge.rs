//! Shared span-propagation helpers for transformation passes.
//!
//! The audit invariant from `spec/design/chelis_span_survival.md` §2.3
//! requires every input Deep span to appear as `span_id` or in
//! `merged_spans` on at least one final IR node. Most passes implement
//! this via the same primitive: append one or more spans to an existing
//! node's `merged_spans`, lex-sorted and deduped, skipping spans already
//! present as `span_id` or in `merged_spans`.
//!
//! These helpers were originally written inline in `lower.rs` for the S2
//! N→1 lowering-collapse rule (rule b). They moved here in S3.0 so other
//! passes (CSE, constant fold, fusion) can apply the same rule without
//! re-implementing the dedup/sort/None-no-op logic.

use crate::dag::{Dag, NodeId};

/// Append a single span to an existing node's `merged_spans`, lex-sorted
/// and deduped. No-op when:
///   * `span` is `None` (passthrough),
///   * the node's `span_id` already equals `span` (already canonical),
///   * `merged_spans` already contains `span` (deduped).
///
/// This is the primitive used by every N→1 merge rule from the per-pass
/// table in §2.3 (lowering's def-collapses-to-body, CSE's duplicate
/// merge, constant fold's operand merge, fusion's contributor merge).
pub fn append_span_to_node(dag: &mut Dag, id: NodeId, span: Option<&str>) {
    let Some(span) = span else {
        return;
    };
    let Some(node) = dag.node_mut(id) else {
        return;
    };
    if node.span_id.as_deref() == Some(span) {
        return;
    }
    if node.merged_spans.iter().any(|s| s == span) {
        return;
    }
    node.merged_spans.push(span.to_owned());
    node.merged_spans.sort();
}

/// Append multiple spans to a node's `merged_spans`, lex-sorted and
/// deduped, with the same no-op rules as `append_span_to_node`. Used by
/// passes that collapse N input nodes into a single survivor and need
/// to atomically transfer N-1 contributors' spans.
pub fn append_spans_to_node(dag: &mut Dag, id: NodeId, spans: &[String]) {
    for span in spans {
        append_span_to_node(dag, id, Some(span.as_str()));
    }
}

/// Merge a duplicate node's full provenance (its `span_id` and existing
/// `merged_spans`) into a survivor node's `merged_spans`. Used by CSE
/// when collapsing structurally-identical nodes onto a survivor: the
/// survivor keeps its own canonical `span_id`, but inherits the
/// duplicate's full audit chain.
///
/// `dup_span_id` is the duplicate's canonical span (or `None` if absent).
/// `dup_merged` is the duplicate's existing `merged_spans` slice.
pub fn merge_duplicate_into_survivor(
    dag: &mut Dag,
    survivor: NodeId,
    dup_span_id: Option<&str>,
    dup_merged: &[String],
) {
    append_span_to_node(dag, survivor, dup_span_id);
    append_spans_to_node(dag, survivor, dup_merged);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{RiscOp, TensorType};

    fn scalar() -> TensorType {
        TensorType::scalar_f32()
    }

    #[test]
    fn append_none_is_noop() {
        let mut dag = Dag::new();
        let id = dag.add_node(
            RiscOp::synth_const(scalar().precision, 0.0),
            vec![],
            scalar(),
            Some("a".into()),
        );
        append_span_to_node(&mut dag, id, None);
        assert_eq!(dag.get(id).unwrap().span_id.as_deref(), Some("a"));
        assert!(dag.get(id).unwrap().merged_spans.is_empty());
    }

    #[test]
    fn append_canonical_is_noop() {
        let mut dag = Dag::new();
        let id = dag.add_node(
            RiscOp::synth_const(scalar().precision, 0.0),
            vec![],
            scalar(),
            Some("a".into()),
        );
        append_span_to_node(&mut dag, id, Some("a"));
        assert!(dag.get(id).unwrap().merged_spans.is_empty());
    }

    #[test]
    fn append_dedup_skips_existing() {
        let mut dag = Dag::new();
        let id = dag.add_node(
            RiscOp::synth_const(scalar().precision, 0.0),
            vec![],
            scalar(),
            Some("a".into()),
        );
        append_span_to_node(&mut dag, id, Some("b"));
        append_span_to_node(&mut dag, id, Some("b"));
        assert_eq!(dag.get(id).unwrap().merged_spans, vec!["b"]);
    }

    #[test]
    fn append_lex_sorts() {
        let mut dag = Dag::new();
        let id = dag.add_node(
            RiscOp::synth_const(scalar().precision, 0.0),
            vec![],
            scalar(),
            Some("a".into()),
        );
        append_span_to_node(&mut dag, id, Some("z"));
        append_span_to_node(&mut dag, id, Some("m"));
        append_span_to_node(&mut dag, id, Some("c"));
        assert_eq!(dag.get(id).unwrap().merged_spans, vec!["c", "m", "z"]);
    }

    #[test]
    fn merge_duplicate_transfers_canonical_and_merged() {
        let mut dag = Dag::new();
        let survivor = dag.add_node(
            RiscOp::synth_const(scalar().precision, 0.0),
            vec![],
            scalar(),
            Some("survivor".into()),
        );
        merge_duplicate_into_survivor(
            &mut dag,
            survivor,
            Some("dup_canonical"),
            &["dup_extra1".into(), "dup_extra2".into()],
        );
        let merged = &dag.get(survivor).unwrap().merged_spans;
        assert_eq!(merged, &vec!["dup_canonical", "dup_extra1", "dup_extra2"]);
    }

    #[test]
    fn merge_duplicate_dedups_against_survivor() {
        let mut dag = Dag::new();
        let survivor = dag.add_node(
            RiscOp::synth_const(scalar().precision, 0.0),
            vec![],
            scalar(),
            Some("a".into()),
        );
        // Survivor's canonical "a" should NOT be re-added.
        merge_duplicate_into_survivor(&mut dag, survivor, Some("a"), &["b".into()]);
        assert_eq!(dag.get(survivor).unwrap().merged_spans, vec!["b"]);
    }
}
