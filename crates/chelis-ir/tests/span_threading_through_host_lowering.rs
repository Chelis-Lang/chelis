//! S6 oracle: span threading through Deep → `HostExpr` host-lane lowering.
//!
//! Mirrors `span_threading_through_lowering.rs` (the S2 DAG-side oracle)
//! but for the host lane. Per `spec/design/chelis_span_survival.md` §2.3
//! host-side rule table, every span ID present on the input Deep AST
//! whose program lowers through the host-lane path must appear as either
//! `span_id` or in `merged_spans` on at least one HostExpr node after
//! `try_lower_compiled_program()`.
//!
//! This file is the named acceptance oracle for S6 steps 1-4. The S6
//! step 5+ phase will extend the audit invariant to backend `// span:`
//! emission; for steps 1-4 the oracle is the in-memory HostExpr tree.
//!
//! Negative-parity tests (CLAUDE.md):
//!  * unspanned input ⇒ all HostExpr nodes carry `span_id = None` and
//!    empty `merged_spans`
//!  * empty program ⇒ does not panic, produces an empty `HostProgram`
//!  * synthesized-marker invariant: any `HostExpr` whose `span_id`
//!    matches `__synthesized_<host_pass>__` MUST have non-empty
//!    `merged_spans` (the synthesized-markers-are-never-the-only-provenance
//!    invariant from S3, locked here even though no shipped pass mints
//!    such markers today).

use std::collections::BTreeSet;

use chelis_deep::Expr;
use chelis_ir::HostTypeTerm;
use chelis_ir::host::{
    CompiledProgram, HostCallback, HostCallbackKind, HostExpr, HostExprKind, HostProgram,
    try_lower_compiled_program,
};
use chelis_types::{CheckedProgram, check_ir_program, check_linearity};

/// Tests in this file provide checked programs that are expected to lower.
/// Keep that expectation explicit while production callers retain the
/// fallible boundary.
fn lower_compiled_program(program: &CheckedProgram) -> CompiledProgram {
    try_lower_compiled_program(program).expect("span fixture must lower through the host boundary")
}

// ── Schema lock test ──────────────────────────────────────────────────

/// Schema-lock regression backstop: `HostExpr` carries BOTH `span_id` and
/// `merged_spans`, mirroring `DagNode`. If a future commit tries to
/// simplify the schema (e.g. drop `merged_spans` because no shipped
/// host-side pass produces N→1 merges today), this test fails the build.
///
/// Per `spec/design/chelis_span_survival.md` §2.3 host-side table, the
/// schema is locked architecturally for orchestrator-tooling uniformity
/// (a single audit consumer reads both DagNode and HostExpr) and for
/// future host-side optimizations that WILL produce merge candidates.
#[test]
fn host_expr_schema_carries_span_id_and_merged_spans() {
    let e = HostExpr::<HostTypeTerm>::new(HostExprKind::Int(0));
    // The struct fields are public — they must stay public so the host
    // emitter (S6 step 5) can read them. If a refactor demotes them to
    // private, this test fails.
    assert!(e.span_id.is_none(), "fresh HostExpr has span_id = None");
    assert!(
        e.merged_spans.is_empty(),
        "fresh HostExpr has empty merged_spans"
    );

    let with_span =
        HostExpr::<HostTypeTerm>::with_span(HostExprKind::Int(1), Some("src_alpha".into()));
    assert_eq!(with_span.span_id.as_deref(), Some("src_alpha"));
    assert!(with_span.merged_spans.is_empty());

    // append_merged_span is the canonical N→1 collapse helper. Lock the
    // dedup / sort / None-noop / canonical-equal-noop semantics so any
    // future refactor is forced to preserve them.
    let mut node = HostExpr::<HostTypeTerm>::with_span(HostExprKind::Int(1), Some("a".into()));
    node.append_merged_span(None);
    assert!(node.merged_spans.is_empty(), "None is a no-op");
    node.append_merged_span(Some("a"));
    assert!(
        node.merged_spans.is_empty(),
        "appending the canonical span_id is a no-op"
    );
    node.append_merged_span(Some("z"));
    node.append_merged_span(Some("c"));
    node.append_merged_span(Some("c"));
    assert_eq!(
        node.merged_spans,
        vec!["c".to_string(), "z".to_string()],
        "merged_spans deduped + lex-sorted"
    );
}

// ── Helpers ────────────────────────────────────────────────────────────

fn check(source: &str) -> chelis_types::CheckedProgram {
    let exprs = chelis_deep::parser::parse_str(source).expect("deep parse");
    let checked = check_ir_program(&exprs).expect("IR check clean");
    let checked = chelis_effects::check_program(&checked).expect("effects clean");
    check_linearity(&checked).expect("linearity clean")
}

fn collect_input_spans(exprs: &[Expr]) -> BTreeSet<String> {
    fn walk(expr: &Expr, acc: &mut BTreeSet<String>) {
        if let Some(s) = expr.span_id() {
            acc.insert(s.to_owned());
        }
        match expr {
            Expr::List(list, _) => list.elements.iter().for_each(|child| walk(child, acc)),
            Expr::Node(node, _) => {
                node.meta()
                    .entries
                    .iter()
                    .for_each(|(_, value)| walk(value, acc));
                node.children_slice()
                    .iter()
                    .for_each(|child| walk(child, acc));
            }
            Expr::Map(map, _) => map.entries.iter().for_each(|(_, value)| walk(value, acc)),
            Expr::MetaExpr(meta, _) => {
                meta.entries.iter().for_each(|(_, value)| walk(value, acc));
                walk(&meta.expr, acc);
            }
            Expr::BareList(elements, _) => elements.iter().for_each(|child| walk(child, acc)),
            Expr::UnknownForm(data) => {
                data.meta
                    .entries
                    .iter()
                    .for_each(|(_, value)| walk(value, acc));
                data.children.iter().for_each(|child| walk(child, acc));
            }
            Expr::Atom(_, _) => {}
        }
    }
    let mut out = BTreeSet::new();
    for expr in exprs {
        walk(expr, &mut out);
    }
    out
}

fn collect_host_program_spans<T>(program: &HostProgram<T>) -> BTreeSet<String> {
    let mut acc = BTreeSet::new();
    for binding in &program.globals {
        collect_host_expr_spans(&binding.value, &mut acc);
    }
    for function in &program.functions {
        collect_host_expr_spans(&function.body, &mut acc);
    }
    acc
}

fn collect_host_expr_spans<T>(expr: &HostExpr<T>, acc: &mut BTreeSet<String>) {
    if let Some(s) = &expr.span_id {
        acc.insert(s.clone());
    }
    for s in &expr.merged_spans {
        acc.insert(s.clone());
    }
    match &expr.kind {
        HostExprKind::List(items, _) | HostExprKind::Tuple(items, _) => {
            for item in items {
                collect_host_expr_spans(item, acc);
            }
        }
        HostExprKind::Call { args, .. } | HostExprKind::Builtin { args, .. } => {
            for arg in args {
                collect_host_expr_spans(arg, acc);
            }
        }
        HostExprKind::AdtConstruct { fields, .. } => {
            for field in fields {
                collect_host_expr_spans(field, acc);
            }
        }
        HostExprKind::AdtFieldAccess { base, .. } => {
            collect_host_expr_spans(base, acc);
        }
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            collect_host_expr_spans(cond, acc);
            collect_host_expr_spans(then_expr, acc);
            collect_host_expr_spans(else_expr, acc);
        }
        HostExprKind::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            collect_host_expr_spans(scrutinee, acc);
            collect_host_expr_spans(some_expr, acc);
            collect_host_expr_spans(none_expr, acc);
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            collect_host_expr_spans(scrutinee, acc);
            for arm in arms {
                collect_host_expr_spans(&arm.expr, acc);
            }
            if let Some(default_expr) = default_expr {
                collect_host_expr_spans(default_expr, acc);
            }
        }
        HostExprKind::Let { bindings, body, .. } => {
            for binding in bindings {
                collect_host_expr_spans(&binding.value, acc);
            }
            collect_host_expr_spans(body, acc);
        }
        HostExprKind::Map { callback, list, .. }
        | HostExprKind::Filter { callback, list, .. }
        | HostExprKind::Partition { callback, list, .. }
        | HostExprKind::FlatMap { callback, list, .. } => {
            collect_host_callback_spans(callback, acc);
            collect_host_expr_spans(list, acc);
        }
        HostExprKind::Fold {
            callback,
            init,
            list,
            ..
        }
        | HostExprKind::Scan {
            callback,
            init,
            list,
            ..
        } => {
            collect_host_callback_spans(callback, acc);
            collect_host_expr_spans(init, acc);
            collect_host_expr_spans(list, acc);
        }
        HostExprKind::TensorCall { args, .. } => {
            for arg in args {
                collect_host_expr_spans(arg, acc);
            }
        }
        _ => {}
    }
}

fn collect_host_callback_spans<T>(callback: &HostCallback<T>, acc: &mut BTreeSet<String>) {
    if let HostCallbackKind::Inline { body, .. } = &callback.kind {
        collect_host_expr_spans(body, acc);
    }
}

fn walk_host_expr<T>(expr: &HostExpr<T>, visit: &mut dyn FnMut(&HostExpr<T>)) {
    visit(expr);
    match &expr.kind {
        HostExprKind::List(items, _) | HostExprKind::Tuple(items, _) => {
            for item in items {
                walk_host_expr(item, visit);
            }
        }
        HostExprKind::Call { args, .. } | HostExprKind::Builtin { args, .. } => {
            for arg in args {
                walk_host_expr(arg, visit);
            }
        }
        HostExprKind::AdtConstruct { fields, .. } => {
            for field in fields {
                walk_host_expr(field, visit);
            }
        }
        HostExprKind::AdtFieldAccess { base, .. } => walk_host_expr(base, visit),
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            walk_host_expr(cond, visit);
            walk_host_expr(then_expr, visit);
            walk_host_expr(else_expr, visit);
        }
        HostExprKind::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            walk_host_expr(scrutinee, visit);
            walk_host_expr(some_expr, visit);
            walk_host_expr(none_expr, visit);
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            walk_host_expr(scrutinee, visit);
            for arm in arms {
                walk_host_expr(&arm.expr, visit);
            }
            if let Some(default_expr) = default_expr {
                walk_host_expr(default_expr, visit);
            }
        }
        HostExprKind::Let { bindings, body, .. } => {
            for binding in bindings {
                walk_host_expr(&binding.value, visit);
            }
            walk_host_expr(body, visit);
        }
        HostExprKind::Map { callback, list, .. }
        | HostExprKind::Filter { callback, list, .. }
        | HostExprKind::Partition { callback, list, .. }
        | HostExprKind::FlatMap { callback, list, .. } => {
            walk_host_callback(callback, visit);
            walk_host_expr(list, visit);
        }
        HostExprKind::Fold {
            callback,
            init,
            list,
            ..
        }
        | HostExprKind::Scan {
            callback,
            init,
            list,
            ..
        } => {
            walk_host_callback(callback, visit);
            walk_host_expr(init, visit);
            walk_host_expr(list, visit);
        }
        HostExprKind::TensorCall { args, .. } => {
            for arg in args {
                walk_host_expr(arg, visit);
            }
        }
        _ => {}
    }
}

fn walk_host_callback<T>(callback: &HostCallback<T>, visit: &mut dyn FnMut(&HostExpr<T>)) {
    if let HostCallbackKind::Inline { body, .. } = &callback.kind {
        walk_host_expr(body, visit);
    }
}

fn walk_host_program<T>(program: &HostProgram<T>, visit: &mut dyn FnMut(&HostExpr<T>)) {
    for binding in &program.globals {
        walk_host_expr(&binding.value, visit);
    }
    for function in &program.functions {
        walk_host_expr(&function.body, visit);
    }
}

// ── Positive parity: span-attributed program lowers cleanly ───────────

/// S6 oracle (named in `spec/design/chelis_span_survival.md` §3 S6 step
/// 4): a span-attributed Deep program that lowers through the host lane
/// produces a `HostProgram` where every input span ID appears as
/// `span_id` or in `merged_spans` on at least one HostExpr node.
///
/// The fixture uses a `(let ...)` over a string-typed binding so the
/// program routes through the host lane (rather than the pure-tensor DAG
/// lane). Every node carries a distinct span so the audit assertion is
/// non-trivial.
#[test]
fn span_threading_through_host_lowering() {
    let source = r#"
        (def {span: "src_def"} top
          (let {span: "src_let"}
            (bind {span: "src_bind"}
              greeting
              (lit {type: (t-prim {} string) span: "src_lit_hello"} "hello"))
            (var {type: (t-prim {} string) span: "src_var_use"} greeting)))
    "#;

    let exprs = chelis_deep::parser::parse_str(source).expect("deep parse");
    let input_spans = collect_input_spans(&exprs);
    let expected: BTreeSet<String> = [
        "src_def",
        "src_let",
        "src_bind",
        "src_lit_hello",
        "src_var_use",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect();
    assert_eq!(
        input_spans, expected,
        "fixture span set drifted (got {input_spans:?})"
    );

    let checked = check(source);
    let compiled = lower_compiled_program(&checked);
    let host_program = compiled
        .host
        .as_ref()
        .expect("string-typed top-level binding routes through host lane");
    let host_spans = collect_host_program_spans(host_program);

    let missing: Vec<&String> = input_spans.difference(&host_spans).collect();
    assert!(
        missing.is_empty(),
        "S6 audit invariant violated: input span IDs {missing:?} did not survive host-lane lowering. \
         Input spans: {input_spans:?}; host spans: {host_spans:?}",
    );

    walk_host_program(host_program, &mut |node| {
        if let Some(s) = &node.span_id {
            assert!(
                input_spans.contains(s),
                "host lowering fabricated a span_id `{s}` not present in input"
            );
        }
        for s in &node.merged_spans {
            assert!(
                input_spans.contains(s),
                "host lowering fabricated a merged_span `{s}` not present in input"
            );
        }
    });
}

// ── Negative parity: no spans on input ⇒ no spans on host ─────────────

/// Negative oracle: a span-free Deep program produces a host tree where
/// every node carries `span_id = None` and empty `merged_spans`. Locks
/// the "no fabrication" rule from the §2.3 host-side table.
#[test]
fn host_lowering_without_spans_produces_none_span_ids() {
    let source = r#"
        (def {} top
          (let {}
            (bind {}
              greeting
              (lit {type: (t-prim {} string)} "hello"))
            (var {type: (t-prim {} string)} greeting)))
    "#;

    let exprs = chelis_deep::parser::parse_str(source).expect("deep parse");
    assert!(
        collect_input_spans(&exprs).is_empty(),
        "negative-parity fixture must have ZERO spans"
    );

    let checked = check(source);
    let compiled = lower_compiled_program(&checked);
    let host_program = compiled
        .host
        .as_ref()
        .expect("string-typed top-level binding routes through host lane");

    walk_host_program(host_program, &mut |node| {
        assert_eq!(
            node.span_id, None,
            "host node `{:?}` carries fabricated span_id despite unspanned input",
            node.kind,
        );
        assert!(
            node.merged_spans.is_empty(),
            "host node `{:?}` fabricated merged_spans `{:?}` despite unspanned input",
            node.kind,
            node.merged_spans,
        );
    });
}

// ── Empty program ⇒ no panic ──────────────────────────────────────────

/// Empty source must not panic and must produce a `CompiledProgram` with
/// no host program. Backward-compat invariant: code that worked before
/// S6 still works.
#[test]
fn host_lowering_an_empty_program_does_not_panic() {
    let source = "";
    let exprs = chelis_deep::parser::parse_str(source).expect("empty deep parse");
    assert!(exprs.is_empty(), "empty source should yield zero exprs");

    let checked = check(source);
    let compiled = lower_compiled_program(&checked);
    assert!(
        compiled.host.is_none() || {
            let program = compiled.host.as_ref().unwrap();
            program.globals.is_empty() && program.functions.is_empty()
        },
        "empty program must lower to None or empty HostProgram",
    );
}

// ── N→1 lowering collapse ─────────────────────────────────────────────

/// Per §2.3 host-side table, rule (b) "N→1 lowering collapse": when a
/// parent Deep expr lowers to a body that already corresponds to an
/// existing HostExpr, the parent's `span_id` appends to the existing
/// node's `merged_spans` (lex-sorted, deduped).
///
/// `(realize {span: "outer"} body)` is the host-side analogue: realize
/// is a host-side identity and returns the body's HostExpr verbatim. The
/// outer realize's span MUST land in the body's `merged_spans` so the
/// audit chain doesn't drop the outer source region.
#[test]
fn realize_wrapper_collapses_into_body_via_merged_spans() {
    let source = r#"
        (def {} top
          (let {}
            (bind {}
              s
              (realize {span: "outer_realize"}
                (lit {type: (t-prim {} string) span: "inner_lit"} "x")))
            (var {type: (t-prim {} string)} s)))
    "#;

    let exprs = chelis_deep::parser::parse_str(source).expect("deep parse");
    let input_spans = collect_input_spans(&exprs);
    assert!(input_spans.contains("outer_realize"));
    assert!(input_spans.contains("inner_lit"));

    let checked = check(source);
    let compiled = lower_compiled_program(&checked);
    let host_program = compiled.host.as_ref().expect("host program");
    let host_spans = collect_host_program_spans(host_program);

    assert!(
        host_spans.contains("outer_realize"),
        "outer realize's span dropped: {host_spans:?}",
    );
    assert!(
        host_spans.contains("inner_lit"),
        "inner lit's span dropped: {host_spans:?}",
    );

    // The String node carries `inner_lit` as canonical span_id and
    // `outer_realize` in merged_spans (the realize collapsed onto it).
    let mut found_collapsed_node = false;
    walk_host_program(host_program, &mut |node| {
        if matches!(&node.kind, HostExprKind::String(_))
            && node.span_id.as_deref() == Some("inner_lit")
            && node.merged_spans.iter().any(|s| s == "outer_realize")
        {
            found_collapsed_node = true;
        }
    });
    assert!(
        found_collapsed_node,
        "expected the inner lit String node to carry `span_id = Some(\"inner_lit\")` \
         AND `merged_spans` containing `outer_realize`; \
         host program: {host_program:#?}",
    );
}

/// Top-level def → body N→1 collapse: `(def {span: a} name body)`
/// produces a `HostBinding` whose `value` is the body's HostExpr. The
/// def's `span_id` MUST append to the value's `merged_spans` (per the
/// §2.3 host-side rule for top-level def collapse).
#[test]
fn top_level_def_collapses_into_body_via_merged_spans() {
    let source = r#"
        (def {span: "outer_def"} top
          (lit {type: (t-prim {} string) span: "inner_str_lit"} "x"))
    "#;

    let checked = check(source);
    let compiled = lower_compiled_program(&checked);
    let host_program = compiled.host.as_ref().expect("host program");
    let binding = host_program
        .globals
        .iter()
        .find(|b| b.name == "top")
        .expect("`top` global");

    // The body's String node carries `inner_str_lit` as canonical and
    // `outer_def` in merged_spans (def collapsed onto body).
    let value = &binding.value;
    assert!(
        matches!(&value.kind, HostExprKind::String(_)),
        "expected value to be a String node, got {:?}",
        value.kind,
    );
    assert_eq!(
        value.span_id.as_deref(),
        Some("inner_str_lit"),
        "expected canonical span_id from the inner lit"
    );
    assert!(
        value.merged_spans.iter().any(|s| s == "outer_def"),
        "expected def's `outer_def` span in merged_spans, got {:?}",
        value.merged_spans,
    );
}

/// Var-ref to a let-bound name: the var-ref's own span must surface on
/// the resulting `HostExpr::Var` node. This is the host-side analogue of
/// the DAG-side cached-binding-lookup test, but the host lane builds a
/// fresh `HostExpr::Var(name, ty)` each time rather than aliasing a
/// cached node — so the var-ref's span lands directly on the new Var
/// node's `span_id`.
#[test]
fn var_ref_to_let_bound_name_carries_var_ref_span() {
    let source = r#"
        (def {} top
          (let {}
            (bind {} a (lit {type: (t-prim {} string) span: "lit_a"} "v"))
            (var {type: (t-prim {} string) span: "var_a_use"} a)))
    "#;

    let checked = check(source);
    let compiled = lower_compiled_program(&checked);
    let host_program = compiled.host.as_ref().expect("host program");
    let host_spans = collect_host_program_spans(host_program);

    assert!(
        host_spans.contains("var_a_use"),
        "var-ref's span `var_a_use` dropped: {host_spans:?}",
    );
    assert!(
        host_spans.contains("lit_a"),
        "lit's span `lit_a` dropped: {host_spans:?}",
    );

    // The var-ref produces a fresh `HostExpr::Var` — locate it.
    let mut found_var_node = false;
    walk_host_program(host_program, &mut |node| {
        if let HostExprKind::Var(name, _) = &node.kind
            && name == "a"
            && node.span_id.as_deref() == Some("var_a_use")
        {
            found_var_node = true;
        }
    });
    assert!(
        found_var_node,
        "expected a `HostExpr::Var` node for `a` carrying `span_id = Some(\"var_a_use\")`; \
         host program: {host_program:#?}",
    );
}

// ── Synthesized-marker invariant ──────────────────────────────────────

/// Synthesized-marker invariant lock: any `HostExpr` whose `span_id`
/// matches `__synthesized_<host_pass>__` MUST have non-empty
/// `merged_spans`. Today's host-side passes do not mint such markers
/// (see §2.3 host-side table notes); this test exists to lock the
/// invariant programmatically so when host-side passes do start
/// producing them, the invariant is enforced from day one.
#[test]
fn synthesized_marker_with_empty_merged_spans_is_invalid() {
    // Simulate a hypothetical host-side pass minting a synthesized
    // marker. The audit invariant from S3 carries to the host side: a
    // synthesized span_id with empty merged_spans is invalid because
    // the audit chain has nothing to resolve back to.
    let invalid_node = HostExpr::with_span(
        HostExprKind::Builtin {
            name: "__some_synthesized__".into(),
            args: Vec::new(),
            ty: HostTypeTerm::Unit,
        },
        Some("__synthesized_future_pass__".into()),
    );
    assert!(
        node_violates_synthesized_marker_invariant(&invalid_node),
        "a HostExpr with synthesized span_id and empty merged_spans \
         must be flagged as invalid by the invariant checker"
    );

    // A synthesized marker with a forward-source span in merged_spans
    // is valid (matches the S3 grad shape: span_id =
    // "__synthesized_grad__", merged_spans = ["forward_node_span"]).
    let mut valid_node = HostExpr::with_span(
        HostExprKind::Builtin {
            name: "__some_synthesized__".into(),
            args: Vec::new(),
            ty: HostTypeTerm::Unit,
        },
        Some("__synthesized_future_pass__".into()),
    );
    valid_node.append_merged_span(Some("forward_source_span"));
    assert!(
        !node_violates_synthesized_marker_invariant(&valid_node),
        "a HostExpr with synthesized span_id AND non-empty merged_spans \
         is valid (carries forward-source provenance)"
    );

    // Real source spans (no `__synthesized_` prefix) trivially satisfy
    // the invariant regardless of merged_spans population.
    let real_node =
        HostExpr::<HostTypeTerm>::with_span(HostExprKind::Int(0), Some("src_alpha".into()));
    assert!(!node_violates_synthesized_marker_invariant(&real_node));
}

/// Scan a HostProgram and assert no node violates the synthesized-marker
/// invariant. This check runs against today's lowered programs and
/// would fail if any host-side pass started minting synthesized markers
/// without forward spans.
#[test]
fn shipped_host_lowering_does_not_violate_synthesized_marker_invariant() {
    let source = r#"
        (def {} top
          (let {}
            (bind {} a (lit {type: (t-prim {} string) span: "lit_a"} "v"))
            (var {type: (t-prim {} string) span: "var_a_use"} a)))
    "#;
    let checked = check(source);
    let compiled = lower_compiled_program(&checked);
    if let Some(host_program) = &compiled.host {
        walk_host_program(host_program, &mut |node| {
            assert!(
                !node_violates_synthesized_marker_invariant(node),
                "shipped host lowering produced a synthesized marker \
                 with empty merged_spans: span_id={:?}, kind={:?}",
                node.span_id,
                node.kind,
            );
        });
    }
}

fn node_violates_synthesized_marker_invariant<T>(node: &HostExpr<T>) -> bool {
    let Some(span) = &node.span_id else {
        return false;
    };
    span.starts_with("__synthesized_") && node.merged_spans.is_empty()
}

// ── Region-corresponding inheritance through let/if/match ─────────────

/// Region-corresponding rule (rule a from §2.3 host-side table): every
/// freshly-produced HostExpr inherits the enclosing Deep expr's span.
/// Test: a span-bearing `(let ...)` lowers to a `HostExprKind::Let`
/// whose canonical `span_id` is the let's own span.
#[test]
fn let_node_inherits_let_exprs_span() {
    let source = r#"
        (def {} top
          (let {span: "outer_let"}
            (bind {} x (lit {type: (t-prim {} string)} "hi"))
            (var {type: (t-prim {} string)} x)))
    "#;
    let checked = check(source);
    let compiled = lower_compiled_program(&checked);
    let host_program = compiled.host.as_ref().expect("host program");
    let binding = host_program
        .globals
        .iter()
        .find(|b| b.name == "top")
        .expect("`top` global");
    let value = &binding.value;
    assert!(
        matches!(&value.kind, HostExprKind::Let { .. }),
        "expected top to be a Let node, got {:?}",
        value.kind,
    );
    assert_eq!(
        value.span_id.as_deref(),
        Some("outer_let"),
        "Let node should carry the let-expr's span as canonical span_id"
    );
}

/// `(if ...)` lowering: the resulting `HostExprKind::If` carries the
/// if-expr's span; the cond/then/else sub-HostExprs carry their own
/// region-corresponding spans.
#[test]
fn if_node_inherits_if_exprs_span() {
    let source = r#"
        (def {} top
          (let {}
            (bind {}
              y
              (if {type: (t-prim {} string) span: "outer_if"}
                  (lit {type: (t-prim {} bool) span: "cond_lit"} true)
                  (lit {type: (t-prim {} string) span: "then_lit"} "yes")
                  (lit {type: (t-prim {} string) span: "else_lit"} "no")))
            (var {type: (t-prim {} string)} y)))
    "#;
    let checked = check(source);
    let compiled = lower_compiled_program(&checked);
    let host_program = compiled.host.as_ref().expect("host program");
    let host_spans = collect_host_program_spans(host_program);
    for s in ["outer_if", "cond_lit", "then_lit", "else_lit"] {
        assert!(host_spans.contains(s), "span `{s}` missing: {host_spans:?}");
    }

    // The If node specifically carries `outer_if`.
    let mut found_if = false;
    walk_host_program(host_program, &mut |node| {
        if matches!(&node.kind, HostExprKind::If { .. })
            && node.span_id.as_deref() == Some("outer_if")
        {
            found_if = true;
        }
    });
    assert!(
        found_if,
        "expected an If HostExpr carrying `span_id = Some(\"outer_if\")`"
    );
}

// ── Programmatic synthesized-marker construction (no code path needed)─

/// The synthesized-marker invariant is checked programmatically (no
/// shipped pass exercises it today). This test provides direct coverage
/// of the helper that detects violations, so the invariant is locked at
/// the type-system / test level even if no production codepath
/// triggers it.
#[test]
fn synthesized_markers_with_provenance_are_well_formed() {
    let mut node = HostExpr::with_span(
        HostExprKind::Var("placeholder".into(), HostTypeTerm::Unit),
        Some("__synthesized_host_pass_a__".into()),
    );
    node.append_merged_span(Some("forward_src"));
    node.append_merged_span(Some("alt_src"));

    assert_eq!(node.span_id.as_deref(), Some("__synthesized_host_pass_a__"));
    assert_eq!(node.merged_spans, vec!["alt_src", "forward_src"]);
    assert!(!node_violates_synthesized_marker_invariant(&node));
}

// ── lower_host_function tensor-helper-extraction span survival ────────
//
// Regression tests for the S6 architectural-foundation red-team's MEDIUM
// finding: `lower_host_function` lowers `body_expr = kids[1]` (the
// inner fn-body) but never observes `body` itself (the surrounding
// `(fn …)` form). Two distinct dropped-span shapes:
//
// (1) `try_lower_tensor_helper_call(&body_expr, …)` returns a
//     `TensorCall` constructed via `HostExpr::new(…)` directly — the
//     `body_expr.span_id()` (the fn-body inner expr's span) drops on
//     the success branch because the wrapper that normally stamps it
//     (`lower_host_expr` at host.rs:~1018) is bypassed.
// (2) The `(fn {span: …} (params …) body_expr)` form's own
//     `meta["span"]` drops on EVERY subpath (success, fallback, and
//     the non-tensor branch) because none of them inspect `body`.
//
// Audit invariant from §2.2: every input Deep span must surface as
// `span_id` or in `merged_spans` on at least one HostExpr node. The
// fix in `lower_host_function` appends both `body_expr.span_id()` and
// `body.span_id()` to `host_body.merged_spans` after the if/else.

/// To exercise `lower_host_function` (host wrapper path), the program
/// needs a `(fn …)` body def AND a host-lane sibling so the
/// `has_any_host_lane_def` gate fires (host.rs:~514, ~556). A
/// string-typed top-level binding satisfies that.
fn fixture_with_tensor_fn(fn_span: &str, body_inner_span: &str) -> String {
    format!(
        r#"
        (def {{}} host_marker
          (lit {{type: (t-prim {{}} string)}} "host"))
        (defsig {{}} double
          (t-fn {{}} (t-tensor {{}} (t-prim {{}} f32)) (t-tensor {{}} (t-prim {{}} f32))))
        (def {{span: "outer_def"}} double
          (fn {{span: "{fn_span}"}}
              (params {{}} (x {{type: (t-tensor {{}} (t-prim {{}} f32))}}))
            (app {{type: (t-tensor {{}} (t-prim {{}} f32)) span: "{body_inner_span}"}}
                 (var {{}} mul)
                 (var {{type: (t-tensor {{}} (t-prim {{}} f32))}} x)
                 (lit {{type: (t-tensor {{}} (t-prim {{}} f32))}} 2.0))))
        "#
    )
}

/// Positive-parity regression: the `(fn …)` form's own meta-span
/// surfaces somewhere in the host program after `lower_host_function`
/// produces a `HostFunction`. This was the dropped-span shape the S6
/// architectural-foundation red-team flagged as MEDIUM.
#[test]
fn fn_form_span_surfaces_through_lower_host_function() {
    let source = fixture_with_tensor_fn("fn_form_span", "body_inner_span");

    let exprs = chelis_deep::parser::parse_str(&source).expect("deep parse");
    let input_spans = collect_input_spans(&exprs);
    assert!(
        input_spans.contains("fn_form_span"),
        "fixture must carry the (fn …) form's span"
    );
    assert!(
        input_spans.contains("body_inner_span"),
        "fixture must carry the fn-body inner expr's span"
    );

    let checked = check(&source);
    let compiled = lower_compiled_program(&checked);
    let host_program = compiled
        .host
        .as_ref()
        .expect("host_marker forces host-lane wrapper for `double`");
    assert!(
        host_program.functions.iter().any(|f| f.name == "double"),
        "expected `double` to lower through `lower_host_function` to host.functions; \
         host program: {host_program:#?}"
    );
    let host_spans = collect_host_program_spans(host_program);

    assert!(
        host_spans.contains("fn_form_span"),
        "(fn …) form's own meta-span dropped: {host_spans:?}",
    );
    assert!(
        host_spans.contains("body_inner_span"),
        "fn-body inner expr's span dropped: {host_spans:?}",
    );
    assert!(
        host_spans.contains("outer_def"),
        "def's outer span dropped (existing host.rs:~581 invariant): {host_spans:?}",
    );

    // No fabrication: all surfaced spans came from the input.
    walk_host_program(host_program, &mut |node| {
        if let Some(s) = &node.span_id {
            assert!(
                input_spans.contains(s),
                "host lowering fabricated a span_id `{s}` not present in input"
            );
        }
        for s in &node.merged_spans {
            assert!(
                input_spans.contains(s),
                "host lowering fabricated a merged_span `{s}` not present in input"
            );
        }
    });
}

/// Negative-parity regression: when the `(fn …)` form has NO span,
/// the lowering does not invent one. The wrapper / lowering must not
/// fabricate a `fn_form_span` where none was given.
#[test]
fn unspanned_fn_form_does_not_fabricate_a_span_in_host_function() {
    let source = r#"
        (def {} host_marker
          (lit {type: (t-prim {} string)} "host"))
        (defsig {} double
          (t-fn {} (t-tensor {} (t-prim {} f32)) (t-tensor {} (t-prim {} f32))))
        (def {} double
          (fn {} (params {} (x {type: (t-tensor {} (t-prim {} f32))}))
            (app {type: (t-tensor {} (t-prim {} f32))}
                 (var {} mul)
                 (var {type: (t-tensor {} (t-prim {} f32))} x)
                 (lit {type: (t-tensor {} (t-prim {} f32))} 2.0))))
    "#;

    let exprs = chelis_deep::parser::parse_str(source).expect("deep parse");
    assert!(
        collect_input_spans(&exprs).is_empty(),
        "fixture must have ZERO spans"
    );

    let checked = check(source);
    let compiled = lower_compiled_program(&checked);
    let host_program = compiled
        .host
        .as_ref()
        .expect("host_marker forces host-lane wrapper for `double`");
    assert!(
        host_program.functions.iter().any(|f| f.name == "double"),
        "expected `double` to lower through `lower_host_function`"
    );

    walk_host_program(host_program, &mut |node| {
        assert_eq!(
            node.span_id, None,
            "host node `{:?}` carries fabricated span_id despite unspanned input",
            node.kind,
        );
        assert!(
            node.merged_spans.is_empty(),
            "host node `{:?}` fabricated merged_spans `{:?}` despite unspanned input",
            node.kind,
            node.merged_spans,
        );
    });
}
