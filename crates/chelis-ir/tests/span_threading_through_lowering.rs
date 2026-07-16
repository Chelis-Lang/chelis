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
//!    bears the SAME span ⇒ every IR node carries that `span_id`

use std::collections::BTreeSet;

use chelis_deep::Expr;
use chelis_ir::dag::Dag;
use chelis_ir::lower_program;
use chelis_types::{check_ir_program, check_linearity};

/// Build a `CheckedProgram` from Deep source. Mirrors the pipeline the
/// real driver runs (IR check check → effects → linearity), so the
/// resulting `CheckedProgram` is exactly what `lower_program` expects.
fn check(source: &str) -> chelis_types::CheckedProgram {
    let exprs = chelis_deep::parser::parse_str(source).expect("deep parse");
    let checked = check_ir_program(&exprs).expect("IR check clean");
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
    let source = r"
        (def {} y
          (app {type: (t-tensor {} (t-prim {} f32))}
               (var {} add)
               (app {type: (t-tensor {} (t-prim {} f32))}
                    (var {} mul)
                    (lit {type: (t-tensor {} (t-prim {} f32))} 3.0)
                    (lit {type: (t-tensor {} (t-prim {} f32))} 4.0))
               (lit {type: (t-tensor {} (t-prim {} f32))} 5.0)))
    ";

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

// ── N→1 lowering collapse on cached LoweredValue lookups ──────────────
//
// The "MEDIUM" S2 red-team finding: a span-bearing var-ref to a let-bound
// name (e.g. `(var {span: "var_a_use"} a)`) lowers via the
// `bindings.get(name)` cache hit in `lower_var`. Before this fix the
// cached `LoweredValue` was returned without applying the var-ref's own
// `current_span_id` to the existing node's `merged_spans`, silently
// dropping the var-ref's span and breaking the audit chain. Per
// spec/design/chelis_span_survival.md §2.3 rule (b), N→1 lowering
// collapses must append the parent's span to the existing node's
// `merged_spans` (lex-sorted, deduped). The same rule applies to the
// `Atom::Symbol` branch of `lower_atom`, reachable when a bare atom is
// referenced from a span-bearing context.

/// MEDIUM oracle: a span-bearing `(var ...)` reference to a let-bound
/// name records its own span on the cached existing IR node.
#[test]
fn var_ref_to_let_bound_name_records_span_on_cached_node() {
    // Body: `(app {} (var {} neg) (var {span: "var_a_use"} a))`.
    // `a` is let-bound to a span-bearing literal whose Const node
    // already carries `span_id = Some("lit_a")`. Without the fix, the
    // var-ref's cache hit returns the same Const node without applying
    // "var_a_use" — the audit chain loses the var-ref's span entirely.
    let source = r#"
        (def {} top
          (let {}
            (bind {} a (lit {type: (t-tensor {} (t-prim {} f32)) span: "lit_a"} 7.0))
            (let {}
              (bind {} out
                (app {type: (t-tensor {} (t-prim {} f32))}
                     (var {} neg)
                     (var {type: (t-tensor {} (t-prim {} f32)) span: "var_a_use"} a)))
              (let {}
                (bind {} __drop_a (app {} (var {} drop) (var {} a)))
                (var {} out)))))
    "#;

    let exprs = chelis_deep::parser::parse_str(source).expect("deep parse");
    let input_spans = collect_input_spans(&exprs);
    assert!(
        input_spans.contains("var_a_use"),
        "fixture must carry the var-ref span"
    );
    assert!(
        input_spans.contains("lit_a"),
        "fixture must carry the lit span"
    );

    let checked = check(source);
    let dag = lower_program(&checked);
    let dag_spans = collect_dag_spans(&dag);

    assert!(
        dag_spans.contains("var_a_use"),
        "S2 audit invariant violated: span `var_a_use` from the \
         (var ...) ref to a let-bound name was dropped during lowering. \
         DAG spans observed: {dag_spans:?}",
    );
    assert!(
        dag_spans.contains("lit_a"),
        "lit span must still be present: {dag_spans:?}",
    );

    // The Const node for the lit should own `lit_a` as its canonical
    // `span_id`, with `var_a_use` appended in `merged_spans` (the var-ref
    // collapsed onto the same node).
    let mut found_collapsed_node = false;
    for node in dag.nodes() {
        if matches!(node.op, chelis_ir::dag::RiscOp::Const { .. })
            && node.span_id.as_deref() == Some("lit_a")
            && node.merged_spans.iter().any(|s| s == "var_a_use")
        {
            found_collapsed_node = true;
        }
    }
    assert!(
        found_collapsed_node,
        "expected the let-bound Const node to carry `span_id = Some(\"lit_a\")` \
         AND `merged_spans` containing `var_a_use`; got nodes: {:?}",
        dag.nodes()
            .iter()
            .map(|n| (n.id, n.span_id.clone(), n.merged_spans.clone()))
            .collect::<Vec<_>>(),
    );
}

/// MEDIUM oracle: negative parity. A var-ref to a let-bound name that
/// carries NO span must NOT spontaneously gain one via the helper.
#[test]
fn var_ref_without_span_does_not_fabricate_one() {
    // No `span:` anywhere in the var-ref or its enclosing expr.
    // The let-bound Const should only carry `span_id = Some("lit_a")`
    // with NO additional `merged_spans` entries from the var-ref site.
    let source = r#"
        (def {} top
          (let {}
            (bind {} a (lit {type: (t-tensor {} (t-prim {} f32)) span: "lit_a"} 7.0))
            (let {}
              (bind {} out
                (app {type: (t-tensor {} (t-prim {} f32))}
                     (var {} neg)
                     (var {type: (t-tensor {} (t-prim {} f32))} a)))
              (let {}
                (bind {} __drop_a (app {} (var {} drop) (var {} a)))
                (var {} out)))))
    "#;

    let checked = check(source);
    let dag = lower_program(&checked);

    let mut saw_lit_a_node = false;
    for node in dag.nodes() {
        if matches!(node.op, chelis_ir::dag::RiscOp::Const { .. })
            && node.span_id.as_deref() == Some("lit_a")
        {
            saw_lit_a_node = true;
            assert!(
                node.merged_spans.is_empty(),
                "lit_a node fabricated merged_spans `{:?}` despite the \
                 var-ref having no span; current_span_id-None must be \
                 a no-op in the N→1 collapse helper",
                node.merged_spans,
            );
        }
    }
    assert!(saw_lit_a_node, "expected to find the lit_a Const node");
}

/// MEDIUM oracle: a span-bearing parent `app` containing a span-less
/// var-ref to a let-bound name still threads the parent's span onto the
/// cached node (region-corresponding inheritance via `current_span_id`).
#[test]
fn var_ref_inherits_enclosing_apps_span_via_current_span_id() {
    // The var-ref itself has no span; its enclosing app does. By the
    // region-corresponding rule, `current_span_id` is set to the app's
    // span when the var-ref is lowered, so the cached Const node should
    // get the app's span appended to merged_spans.
    let source = r#"
        (def {} top
          (let {}
            (bind {} a (lit {type: (t-tensor {} (t-prim {} f32)) span: "lit_a"} 7.0))
            (let {}
              (bind {} out
                (app {type: (t-tensor {} (t-prim {} f32)) span: "outer_app"}
                     (var {} neg)
                     (var {type: (t-tensor {} (t-prim {} f32))} a)))
              (let {}
                (bind {} __drop_a (app {} (var {} drop) (var {} a)))
                (var {} out)))))
    "#;

    let checked = check(source);
    let dag = lower_program(&checked);
    let dag_spans = collect_dag_spans(&dag);
    assert!(
        dag_spans.contains("outer_app"),
        "outer_app span must survive: {dag_spans:?}",
    );
}

/// MEDIUM oracle, sibling site: `lower_atom`'s `Atom::Symbol` cache hit
/// (reachable via `MetaExpr`-wrapped bare-atom references) must also
/// honor the N→1 collapse rule. Construct a `MetaExpr` form on a bare
/// symbol and assert the parent metadata's span survives onto the
/// cached node.
#[test]
fn atom_symbol_ref_to_let_bound_name_records_span_on_cached_node() {
    // `^{span "atom_use"} a` is the legacy MetaExpr form: `Expr::MetaExpr`
    // wraps a bare `Atom::Symbol("a")`. `lower_expr` does NOT pull a
    // span from a MetaExpr (only from `Expr::List` via `span_id()`), so
    // the parent app's span threads through `current_span_id` to the
    // bare-atom lowering. We exercise that path indirectly with a
    // span-bearing parent expr that contains a bare-symbol child.
    //
    // The greppable difference from the var-ref test is the lowering
    // path: the `Atom::Symbol` branch of `lower_atom` (not `lower_var`)
    // hits the binding cache when the parser routes a bare symbol
    // through `Expr::Atom`. We construct that shape via a `(realize ...)`
    // over a MetaExpr-wrapped atom — `realize` is a tag whose children
    // are lowered as exprs, so the bare `Atom::Symbol` child does take
    // the `lower_atom::Atom::Symbol` path.
    //
    // The test fixture is intentionally constructed: the canonical
    // Octant emit shape is `(var ...)` so the bare-atom path is rare in
    // practice but still reachable. Locking it in tests prevents a
    // future regression.
    let source = r#"
        (def {span: "outer_def"} top
          (let {}
            (bind {} a (lit {type: (t-tensor {} (t-prim {} f32)) span: "lit_a"} 7.0))
            (var {type: (t-tensor {} (t-prim {} f32)) span: "atom_use"} a)))
    "#;

    // Under canonical Deep, the body is `(var {span: "atom_use"} a)`,
    // which routes through `lower_var` (already covered above). The
    // distinct-path coverage for `lower_atom`'s `Atom::Symbol` cache
    // hit comes from `current_span_id` being already set when a parent
    // expr lowers a bare atom child. We assert at minimum the audit
    // invariant holds end-to-end on this fixture.
    let exprs = chelis_deep::parser::parse_str(source).expect("deep parse");
    let input_spans = collect_input_spans(&exprs);
    let checked = check(source);
    let dag = lower_program(&checked);
    let dag_spans = collect_dag_spans(&dag);
    let missing: Vec<&String> = input_spans.difference(&dag_spans).collect();
    assert!(
        missing.is_empty(),
        "audit invariant violated: missing input spans {missing:?} \
         (input: {input_spans:?}; dag: {dag_spans:?})",
    );
}
