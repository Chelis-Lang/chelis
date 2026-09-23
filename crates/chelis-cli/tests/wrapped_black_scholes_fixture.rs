//! S5.0 fixture verification — the wrapped Black-Scholes Deep fixture
//! must (a) parse cleanly, (b) round-trip through `chelis fmt`, (c)
//! typecheck via `chelis_types::check_ir_program`, and (d) carry
//! the expected span IDs end-to-end.
//!
//! `call_price.dp` (the Octant-emitted equation-only fixture) and
//! `call_price_wrapped.dp` (the typecheckable wrapper, mostly produced
//! by Octant's `--wrap-as-function` translation surface as of Octant
//! commit 468bdc6) are both kept under
//! `crates/chelis-cli/tests/fixtures/octant/black_scholes/`. The
//! wrapper inlines the Octant-emitted bodies as `let` bindings inside
//! a single function def so the program is self-contained and
//! typecheckable; see the fixture README for the regeneration recipe
//! and the rationale for keeping a local `normal_cdf` stub instead of
//! Octant's natural `Nautilus.Special.normal_cdf` access chain.

use chelis_deep::Expr;
use chelis_deep::parser::{parse_str, parse_str_strict};
use chelis_deep::printer::print_canonical;
use std::collections::BTreeSet;

const WRAPPED_DP: &str = include_str!("fixtures/octant/black_scholes/call_price_wrapped.dp");
const WRAPPED_SPANS_JSON: &str =
    include_str!("fixtures/octant/black_scholes/call_price_wrapped.spans.json");

fn collect_span_ids(exprs: &[Expr]) -> Vec<String> {
    let mut acc = Vec::new();
    for expr in exprs {
        collect_one(expr, &mut acc);
    }
    acc
}

fn collect_one(expr: &Expr, acc: &mut Vec<String>) {
    if let Some(id) = expr.span_id() {
        acc.push(id.to_string());
    }
    match expr {
        Expr::Atom(..) => {}
        Expr::Map(meta, _) => {
            meta.visit_syntax(&mut |_, value| collect_one(value, acc));
        }
        Expr::MetaExpr(meta, _) => {
            meta.metadata
                .visit_syntax(&mut |_, value| collect_one(value, acc));
            collect_one(&meta.expr, acc);
        }
        Expr::Node(node, _) => {
            node.meta()
                .visit_syntax(&mut |_, value| collect_one(value, acc));
            for child in node.children_slice() {
                collect_one(child, acc);
            }
        }
        Expr::BareList(elements, _) => {
            for child in elements {
                collect_one(child, acc);
            }
        }
        Expr::UnknownForm(data) => {
            data.meta
                .visit_syntax(&mut |_, value| collect_one(value, acc));
            for child in &data.children {
                collect_one(child, acc);
            }
        }
    }
}

#[test]
fn wrapped_fixture_parses_strict() {
    parse_str_strict(WRAPPED_DP).expect("wrapped fixture must parse strictly");
}

#[test]
fn wrapped_fixture_round_trips_through_fmt() {
    let exprs1 = parse_str(WRAPPED_DP).expect("first parse");
    let printed = print_canonical(&exprs1);
    let exprs2 = parse_str(&printed).expect("re-parse after fmt");
    let reprinted = print_canonical(&exprs2);
    assert_eq!(
        printed, reprinted,
        "fmt must be a fixed point on the wrapped fixture"
    );
    let ids1 = collect_span_ids(&exprs1);
    let ids2 = collect_span_ids(&exprs2);
    assert_eq!(
        ids1, ids2,
        "round-trip through fmt must preserve every span ID in order"
    );
}

#[test]
fn wrapped_fixture_typechecks_ir() {
    let exprs = parse_str_strict(WRAPPED_DP).expect("parse");
    match chelis_types::check_ir_program(&exprs) {
        Ok(_) => {}
        Err(report) => {
            let msgs: Vec<String> = report
                .errors
                .iter()
                .map(|e| format!("{:?}: {}", e.kind, e.message))
                .collect();
            panic!(
                "wrapped Black-Scholes fixture must IR check typecheck; errors:\n  {}",
                msgs.join("\n  ")
            );
        }
    }
}

#[test]
fn wrapped_fixture_span_ids_match_sidecar() {
    // Spec contract: every span ID present in the .dp must appear in
    // the .spans.json sidecar (the audit chain), and vice versa.
    let exprs = parse_str(WRAPPED_DP).expect("parse");
    let dp_ids: BTreeSet<String> = collect_span_ids(&exprs).into_iter().collect();

    // Parse the .spans.json minimally — extract every "deep_node_id"
    // value via serde_json without pulling in a full schema.
    let v: serde_json::Value = serde_json::from_str(WRAPPED_SPANS_JSON).expect("sidecar JSON");
    let arr = v
        .get("spans")
        .and_then(|s| s.as_array())
        .expect("sidecar has `spans` array");
    let sidecar_ids: BTreeSet<String> = arr
        .iter()
        .filter_map(|s| {
            s.get("deep_node_id")
                .and_then(|d| d.as_str())
                .map(str::to_string)
        })
        .collect();

    assert_eq!(
        dp_ids,
        sidecar_ids,
        "every span ID in call_price_wrapped.dp must appear in \
         call_price_wrapped.spans.json (and vice versa); dp_only={:?}, \
         sidecar_only={:?}",
        dp_ids.difference(&sidecar_ids).collect::<Vec<_>>(),
        sidecar_ids.difference(&dp_ids).collect::<Vec<_>>(),
    );
}

#[test]
fn wrapped_fixture_includes_octant_emitted_spans() {
    // Sanity: the bodies inlined into the wrapper carry the verbatim
    // Octant-emitted span IDs (eq:d1_*, eq:d2_*, eq:c_*). If the
    // fixture is regenerated from a different .tex, this list updates.
    let exprs = parse_str(WRAPPED_DP).expect("parse");
    let ids: BTreeSet<String> = collect_span_ids(&exprs).into_iter().collect();
    for octant_id in [
        "eq:d1_002",
        "eq:d1_006",
        "eq:d1_017",
        "eq:d2_003",
        "eq:c_002",
        "eq:c_020",
    ] {
        assert!(
            ids.contains(octant_id),
            "wrapped fixture must inline the Octant-emitted span {octant_id} \
             (got: {ids:?})"
        );
    }
    // And the canonical synthesized-wrap marker (spec/03-deep-syntax.md
    // §1.1.1: reserved `__synthesized_<pass>__` form). Both wrapper
    // defs in the fixture share this single canonical marker, matching
    // Octant's `--wrap-as-function` natural output (commit 468bdc6).
    assert!(
        ids.contains("__synthesized_wrap__"),
        "wrapped fixture must mark wrapper defs with the canonical \
         `__synthesized_wrap__` marker (got: {ids:?})"
    );
}
