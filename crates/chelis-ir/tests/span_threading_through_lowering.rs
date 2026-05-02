//! S2 oracle: span threading through Deep → IR lowering.
//!
//! Per `spec/design/chelis_span_survival.md` §3 (S2): every span ID
//! present on the input Deep AST must appear as the `span_id` of at
//! least one IR node after `lower_program()`. Nodes that have no source
//! region carry `span_id = None` (the audit invariant only fires on
//! input spans, not on synthesized lowering artifacts).
//!
//! This file is the named acceptance oracle for S2. The S3 phase will
//! extend the audit invariant to the post-optimization-pass DAG; for S2
//! the only pass that runs after lowering is DCE, which is a pure copy
//! per design §2.3.
//!
//! Negative-parity tests (CLAUDE.md):
//!  * unspanned input ⇒ all IR nodes carry `span_id = None`
//!  * empty program ⇒ doesn't panic, produces empty DAG
//!  * single-source-region invariance: a Deep program where every node
//!    bears the SAME span ⇒ every IR node carries that span_id

use std::collections::BTreeSet;

use chelis_deep::Expr;
use chelis_ir::dag::Dag;
use chelis_ir::lower_program;
use chelis_types::{check_linearity, check_phase0e_program};

/// Build a CheckedProgram from Deep source. Mirrors the pipeline the
/// real driver runs (Phase 0e check → effects → linearity), so the
/// resulting `CheckedProgram` is exactly what `lower_program` expects.
fn check(source: &str) -> chelis_types::CheckedProgram {
    let exprs = chelis_deep::parser::parse_str(source).expect("deep parse");
    let checked = check_phase0e_program(&exprs).expect("Phase 0e clean");
    let checked = chelis_effects::check_program(&checked).expect("effects clean");
    check_linearity(&checked).expect("linearity clean")
}

/// Walk the Deep AST and collect every span ID present.
fn collect_input_spans(exprs: &[Expr]) -> BTreeSet<String> {
    fn walk(expr: &Expr, acc: &mut BTreeSet<String>) {
        if let Some(s) = expr.span_id() {
            acc.insert(s.to_owned());
        }
        if let Expr::List(list, _) = expr {
            for child in &list.elements {
                walk(child, acc);
            }
        }
    }
    let mut out = BTreeSet::new();
    for expr in exprs {
        walk(expr, &mut out);
    }
    out
}

/// Walk the lowered DAG and collect every span ID present on any node,
/// pulling from BOTH `span_id` and `merged_spans`. Empty strings are
/// valid span IDs and are kept; only `None` `span_id`s are excluded.
fn collect_dag_spans(dag: &Dag) -> BTreeSet<String> {
    let mut acc = BTreeSet::new();
    for n in dag.nodes() {
        if let Some(s) = &n.span_id {
            acc.insert(s.clone());
        }
        for s in &n.merged_spans {
            acc.insert(s.clone());
        }
    }
    acc
}

// ── Positive: spans on real input flow into IR ────────────────────────

/// S2 oracle (named in `spec/design/chelis_span_survival.md` §3 S2):
/// `lower_program` on a span-attributed Deep program produces an IR
/// where every input span ID appears on at least one IR node.
#[test]
fn span_threading_through_lowering() {
    // A small but realistic program exercising binding + arithmetic.
    // Each `(app …)` and most leaves carry their own span; the span IDs
    // are designed to be distinct so the test's "every input span
    // appears in IR" assertion is non-trivial.
    // Mirror the phase_f parity_pair_1 fixture so we know it lowers cleanly,
    // then layer span metadata onto every node that the parser threads
    // metadata through. `mul` and `add` are builtins (BUILTIN_NAMES), so
    // no library context is required for this program to type-check + lower.
    let source = r#"
        (def {span: "src_def"} y
          (app {type: (t-tensor {} (t-prim {} f32)) span: "src_root_app"}
               (var {} add)
               (app {type: (t-tensor {} (t-prim {} f32)) span: "src_inner_mul"}
                    (var {} mul)
                    (lit {type: (t-tensor {} (t-prim {} f32)) span: "src_lit_a"} 3.0)
                    (lit {type: (t-tensor {} (t-prim {} f32)) span: "src_lit_b"} 4.0))
               (lit {type: (t-tensor {} (t-prim {} f32)) span: "src_lit_c"} 5.0)))
    "#;

    let exprs = chelis_deep::parser::parse_str(source).expect("deep parse");
    let input_spans = collect_input_spans(&exprs);
    assert!(
        !input_spans.is_empty(),
        "test fixture must carry at least one span"
    );
    // Sanity: lock the expected set so a future fixture edit can't
    // silently weaken the assertion.
    let expected: BTreeSet<String> = [
        "src_def",
        "src_root_app",
        "src_inner_mul",
        "src_lit_a",
        "src_lit_b",
        "src_lit_c",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect();
    assert_eq!(
        input_spans, expected,
        "fixture span set drifted (got {input_spans:?})"
    );

    let checked = check(source);
    let dag = lower_program(&checked);
    let dag_spans = collect_dag_spans(&dag);
    let missing: Vec<&String> = input_spans.difference(&dag_spans).collect();
    assert!(
        missing.is_empty(),
        "S2 audit invariant violated: input span IDs {missing:?} did not survive lowering. \
         Input spans: {input_spans:?}; DAG spans: {dag_spans:?}",
    );

    // Every node's span_id is either `None` (legitimately synthesized
    // with no source region) or a string drawn from the input span set
    // (i.e. lowering doesn't fabricate span IDs). The same constraint
    // applies to merged_spans entries.
    for node in dag.nodes() {
        if let Some(s) = &node.span_id {
            assert!(
                input_spans.contains(s),
                "lowering fabricated a span ID `{s}` not present in input"
            );
        }
        for s in &node.merged_spans {
            assert!(
                input_spans.contains(s),
                "lowering fabricated a merged_span `{s}` not present in input"
            );
        }
    }
}

// ── Negative: no spans on input ⇒ no spans on IR ─────────────────────

#[test]
fn lowering_without_spans_produces_none_span_ids() {
    // Same shape as the positive test, but no `span:` keys anywhere.
    // Every IR node must have `span_id = None` — lowering must not
    // fabricate spans for unspanned input.
    let source = r#"
        (def {} y
          (app {type: (t-tensor {} (t-prim {} f32))}
               (var {} add)
               (app {type: (t-tensor {} (t-prim {} f32))}
                    (var {} mul)
                    (lit {type: (t-tensor {} (t-prim {} f32))} 3.0)
                    (lit {type: (t-tensor {} (t-prim {} f32))} 4.0))
               (lit {type: (t-tensor {} (t-prim {} f32))} 5.0)))
    "#;

    let exprs = chelis_deep::parser::parse_str(source).expect("deep parse");
    assert!(
        collect_input_spans(&exprs).is_empty(),
        "negative-parity fixture must have ZERO spans"
    );

    let checked = check(source);
    let dag = lower_program(&checked);

    assert!(
        !dag.is_empty(),
        "negative-parity fixture should still produce a non-empty DAG"
    );
    for node in dag.nodes() {
        assert_eq!(
            node.span_id, None,
            "node {:?} carries fabricated span_id `{:?}` despite unspanned input",
            node.id, node.span_id,
        );
        assert!(
            node.merged_spans.is_empty(),
            "node {:?} fabricated merged_spans `{:?}` despite unspanned input",
            node.id,
            node.merged_spans,
        );
    }
}

// ── Negative: empty program ⇒ no panic, empty DAG ─────────────────────

#[test]
fn lowering_an_empty_program_does_not_panic() {
    // No top-level decls. The program must not panic and must produce
    // an empty (zero-node) DAG. Backward-compat invariant: code that
    // worked before S2 still works.
    let source = "";
    let exprs = chelis_deep::parser::parse_str(source).expect("empty deep parse");
    assert!(exprs.is_empty(), "empty source should yield zero exprs");

    let checked = check(source);
    let dag = lower_program(&checked);
    assert!(
        dag.is_empty(),
        "empty program must lower to an empty DAG (got {} nodes)",
        dag.len()
    );
}

// ── Single-source-region invariance ───────────────────────────────────

#[test]
fn single_span_program_propagates_to_every_ir_node() {
    // Every node carries the SAME span. By the region-corresponding
    // rule (every IR node lowered from a child without its own span
    // inherits the parent's), every IR node should carry `span_uniform`.
    // This test catches regressions where an inner helper drops the
    // current_span_id thread (e.g. by passing None to add_node when
    // current_span_id is Some).
    let source = r#"
        (def {span: "span_uniform"} y
          (app {type: (t-tensor {} (t-prim {} f32)) span: "span_uniform"}
               (var {} add)
               (lit {type: (t-tensor {} (t-prim {} f32)) span: "span_uniform"} 1.0)
               (lit {type: (t-tensor {} (t-prim {} f32)) span: "span_uniform"} 2.0)))
    "#;
    let checked = check(source);
    let dag = lower_program(&checked);

    assert!(!dag.is_empty(), "uniform-span fixture should lower");
    let mut saw_uniform = false;
    for node in dag.nodes() {
        match node.span_id.as_deref() {
            Some("span_uniform") => saw_uniform = true,
            Some(other) => panic!(
                "node {:?} has unexpected span_id `{other}`; only `span_uniform` should appear",
                node.id
            ),
            None => panic!(
                "node {:?} carries `span_id = None` despite every input node having a span; \
                 region-corresponding inheritance is broken",
                node.id
            ),
        }
    }
    assert!(saw_uniform, "no IR node carried the uniform span");
}

// ── Helper-pass invariant: synthesized markers are S3 scope ───────────

#[test]
fn s2_lowering_does_not_emit_synthesized_markers() {
    // S2 design states: only `__synthesized_tier2__` (S3) and
    // `__synthesized_grad__` (S3) are valid synthesized markers, and
    // only S3 passes mint them. S2 lowering must NOT emit any
    // `__synthesized_*__` strings — if the lowering rules need a
    // synthesized marker, that's a design gap (escalate per CLAUDE.md).
    let source = r#"
        (def {span: "u_def"} y
          (app {type: (t-tensor {} (t-prim {} f32)) span: "u_app"}
               (var {} neg)
               (lit {type: (t-tensor {} (t-prim {} f32)) span: "u_lit"} 7.0)))
    "#;
    let checked = check(source);
    let dag = lower_program(&checked);

    for node in dag.nodes() {
        if let Some(s) = &node.span_id {
            assert!(
                !s.starts_with("__synthesized_"),
                "S2 lowering minted synthesized marker `{s}` on node {:?}; \
                 markers are S3 scope per spec/design/chelis_span_survival.md §2.3",
                node.id,
            );
        }
        for s in &node.merged_spans {
            assert!(
                !s.starts_with("__synthesized_"),
                "S2 lowering minted synthesized marker `{s}` on node {:?} merged_spans",
                node.id,
            );
        }
    }
}

// ── Atom literals follow the parent's region-corresponding rule ───────

#[test]
fn atom_const_inherits_parent_span() {
    // `Atom::Float(7.0)` has no metadata of its own. It should inherit
    // the parent `lit`'s span via current_span_id threading.
    let source = r#"
        (def {span: "atom_def"} y
          (lit {type: (t-tensor {} (t-prim {} f32)) span: "atom_lit"} 7.0))
    "#;
    let checked = check(source);
    let dag = lower_program(&checked);

    let mut found_atom_span = false;
    for node in dag.nodes() {
        if matches!(
            node.op,
            chelis_ir::dag::RiscOp::Const { .. } | chelis_ir::dag::RiscOp::Store { .. }
        ) && node.span_id.as_deref() == Some("atom_lit")
        {
            found_atom_span = true;
        }
    }
    assert!(
        found_atom_span,
        "expected at least one Const/Store node carrying span `atom_lit` \
         (region-corresponding inheritance from the parent `lit` expr)"
    );
}
